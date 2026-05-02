//! Stage 2, the per-layer eigenmode analysis.
//!
//! Implements PP2017 Eq. (11) for the eigenvalue problem,
//! Eq. (12) for the forward-backward partition, Eqs. (13) and (14)
//! for the Li-Sullivan-Parsons sort, Eqs. (15) and (16) for the
//! Poynting-vector fallback in birefringent media, and Eqs. (17) and (18)
//! for the longitudinal `E_z, H_z` recovery. See
//! `docs/theory/eigenmode_analysis.md`.

use crate::error::{KernelError, KernelResult};
use crate::kernel::delta::DeltaMatrix;
use crate::types::scalar::C64;
use nalgebra::{Matrix4, Vector4};

/// Output of the eigenmode-and-sort pass for one layer.
#[derive(Debug, Clone, Copy)]
pub struct LayerModes {
    /// Sorted eigenvalues `(q_i1, q_i2, q_i3, q_i4)` in the
    /// `(p-trans, s-trans, p-refl, s-refl)` Passler convention.
    pub q: [C64; 4],
    /// Corresponding right eigenvectors in column-major layout,
    /// each column carrying one of the four 4-vectors `Psi_ij`.
    pub psi: Matrix4<C64>,
}

/// Forward-backward classification for a single eigenvalue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Transmitted,
    Reflected,
}

const IM_EQ_TOL: f64 = 1.0e-12;
const LI_TOL: f64 = 1.0e-12;

fn zgeev_right(delta: &DeltaMatrix) -> KernelResult<([C64; 4], Matrix4<C64>)> {
    let n = 4i32;
    let lda = n;
    let mut a = *delta;
    let mut w = vec![C64::default(); 4];
    let mut vl = vec![C64::default(); 1];
    let ldvl = 1i32;
    let mut vr = vec![C64::default(); 16];
    let ldvr = n;
    let mut rwork = vec![0.0f64; 2 * n as usize];
    let mut info = 0i32;
    let mut work = vec![C64::default(); 1];
    let mut lwork = -1i32;
    unsafe {
        lapack::zgeev(
            b'N',
            b'V',
            n,
            a.as_mut_slice(),
            lda,
            &mut w,
            &mut vl,
            ldvl,
            &mut vr,
            ldvr,
            &mut work,
            lwork,
            &mut rwork,
            &mut info,
        );
    }
    if info != 0 {
        return Err(KernelError::EigenSolveFailure {
            layer_index: 0,
            message: format!("zgeev workspace query failed info={info}"),
        });
    }
    lwork = work[0].re as i32;
    if lwork < 1 {
        lwork = 4 * n;
    }
    work.resize(lwork as usize, C64::default());
    a = *delta;
    unsafe {
        lapack::zgeev(
            b'N',
            b'V',
            n,
            a.as_mut_slice(),
            lda,
            &mut w,
            &mut vl,
            ldvl,
            &mut vr,
            ldvr,
            &mut work,
            lwork,
            &mut rwork,
            &mut info,
        );
    }
    if info != 0 {
        return Err(KernelError::EigenSolveFailure {
            layer_index: 0,
            message: format!("zgeev failed info={info}"),
        });
    }
    let mut psi = Matrix4::zeros();
    for j in 0..4 {
        for i in 0..4 {
            psi[(i, j)] = vr[i + j * 4];
        }
    }
    Ok(([w[0], w[1], w[2], w[3]], psi))
}

/// Solve the eigenvalue problem `q Psi = Delta Psi` of PP2017
/// Eq. (11) with a dense complex eigensolver.
pub fn solve_eigenmodes(delta: &DeltaMatrix) -> KernelResult<([C64; 4], Matrix4<C64>)> {
    zgeev_right(delta)
}

fn classify_one(q: C64) -> KernelResult<Direction> {
    if q.im.abs() > IM_EQ_TOL {
        if q.im >= 0.0 {
            Ok(Direction::Transmitted)
        } else {
            Ok(Direction::Reflected)
        }
    } else if q.re.abs() > IM_EQ_TOL {
        if q.re >= 0.0 {
            Ok(Direction::Transmitted)
        } else {
            Ok(Direction::Reflected)
        }
    } else {
        Err(KernelError::AmbiguousPartition { layer_index: 0 })
    }
}

/// Apply PP2017 Eq. (12) to classify each eigenvalue as
/// transmitted or reflected.
pub fn partition_modes(q: &[C64; 4]) -> KernelResult<[Direction; 4]> {
    Ok([
        classify_one(q[0])?,
        classify_one(q[1])?,
        classify_one(q[2])?,
        classify_one(q[3])?,
    ])
}

fn c_electric(col: &Vector4<C64>) -> f64 {
    let ex = col[0];
    let ey = col[2];
    let nx = ex.norm_sqr();
    let ny = ey.norm_sqr();
    let d = nx + ny;
    if d <= 1.0e-30 {
        return 0.5;
    }
    nx / d
}

fn c_poynting(col: &Vector4<C64>, a3n: &[C64; 6], a6n: &[C64; 6]) -> KernelResult<f64> {
    let ex = col[0];
    let hy = col[1];
    let ey = col[2];
    let nhx = -col[3];
    let hx = -nhx;
    let (ez, hz) = longitudinal_components(col, a3n, a6n)?;
    let sx = ey * hz - ez * hy;
    let sy = ez * hx - ex * hz;
    let nx: f64 = sx.norm_sqr();
    let ny: f64 = sy.norm_sqr();
    let d = nx + ny;
    if d <= 1.0e-30 {
        return Ok(0.5);
    }
    Ok(nx / d)
}

fn order_two(
    idx: [usize; 2],
    _q: &[C64; 4],
    psi: &Matrix4<C64>,
    dir_ok: impl Fn(usize) -> bool,
    use_poynting: bool,
    a3n: &[C64; 6],
    a6n: &[C64; 6],
    layer_index: usize,
) -> KernelResult<[usize; 2]> {
    let i0 = idx[0];
    let i1 = idx[1];
    if !dir_ok(i0) || !dir_ok(i1) {
        return Err(KernelError::InvalidGeometry(
            "partition mismatch in pair ordering".into(),
        ));
    }
    let c0 = if use_poynting {
        c_poynting(&psi.column(i0).into_owned(), a3n, a6n)?
    } else {
        c_electric(&psi.column(i0).into_owned())
    };
    let c1 = if use_poynting {
        c_poynting(&psi.column(i1).into_owned(), a3n, a6n)?
    } else {
        c_electric(&psi.column(i1).into_owned())
    };
    if (c0 - c1).abs() < LI_TOL {
        return Err(KernelError::AmbiguousLiSort {
            layer_index,
            threshold: LI_TOL,
        });
    }
    if c0 > c1 {
        Ok([i0, i1])
    } else {
        Ok([i1, i0])
    }
}

/// Apply the Li-Sullivan-Parsons electric-projection sort of
/// PP2017 Eqs. (13) and (14).
pub fn sort_li(
    q: &[C64; 4],
    psi: &Matrix4<C64>,
    direction: &[Direction; 4],
    layer_index: usize,
) -> KernelResult<LayerModes> {
    let tr: Vec<usize> = (0..4).filter(|&j| direction[j] == Direction::Transmitted).collect();
    let rf: Vec<usize> = (0..4).filter(|&j| direction[j] == Direction::Reflected).collect();
    if tr.len() != 2 || rf.len() != 2 {
        return Err(KernelError::InvalidGeometry(
            "expected two transmitted and two reflected modes".into(),
        ));
    }
    let tr_pair = order_two(
        [tr[0], tr[1]],
        q,
        psi,
        |j| direction[j] == Direction::Transmitted,
        false,
        &[C64::default(); 6],
        &[C64::default(); 6],
        layer_index,
    )?;
    let rf_pair = order_two(
        [rf[0], rf[1]],
        q,
        psi,
        |j| direction[j] == Direction::Reflected,
        false,
        &[C64::default(); 6],
        &[C64::default(); 6],
        layer_index,
    )?;
    let idx = [tr_pair[0], tr_pair[1], rf_pair[0], rf_pair[1]];
    let mut qo = [C64::default(); 4];
    let mut psio = Matrix4::zeros();
    for (k, &j) in idx.iter().enumerate() {
        qo[k] = q[j];
        psio.set_column(k, &psi.column(j));
    }
    Ok(LayerModes {
        q: qo,
        psi: psio,
    })
}

/// Apply the Poynting-vector fallback sort of PP2017 Eqs. (15)
/// and (16) for the birefringent regime.
pub fn sort_poynting(
    q: &[C64; 4],
    psi: &Matrix4<C64>,
    direction: &[Direction; 4],
    a3n: &[C64; 6],
    a6n: &[C64; 6],
    layer_index: usize,
) -> KernelResult<LayerModes> {
    let tr: Vec<usize> = (0..4).filter(|&j| direction[j] == Direction::Transmitted).collect();
    let rf: Vec<usize> = (0..4).filter(|&j| direction[j] == Direction::Reflected).collect();
    if tr.len() != 2 || rf.len() != 2 {
        return Err(KernelError::InvalidGeometry(
            "expected two transmitted and two reflected modes".into(),
        ));
    }
    let tr_pair = order_two(
        [tr[0], tr[1]],
        q,
        psi,
        |j| direction[j] == Direction::Transmitted,
        true,
        a3n,
        a6n,
        layer_index,
    )?;
    let rf_pair = order_two(
        [rf[0], rf[1]],
        q,
        psi,
        |j| direction[j] == Direction::Reflected,
        true,
        a3n,
        a6n,
        layer_index,
    )?;
    let idx = [tr_pair[0], tr_pair[1], rf_pair[0], rf_pair[1]];
    let mut qo = [C64::default(); 4];
    let mut psio = Matrix4::zeros();
    for (k, &j) in idx.iter().enumerate() {
        qo[k] = q[j];
        psio.set_column(k, &psi.column(j));
    }
    Ok(LayerModes {
        q: qo,
        psi: psio,
    })
}

/// Apply Li sort with automatic Poynting fallback when the electric
/// projection is numerically degenerate.
pub fn sort_modes_with_fallback(
    q: &[C64; 4],
    psi: &Matrix4<C64>,
    direction: &[Direction; 4],
    a3n: &[C64; 6],
    a6n: &[C64; 6],
    layer_index: usize,
) -> KernelResult<LayerModes> {
    match sort_li(q, psi, direction, layer_index) {
        Ok(m) => Ok(m),
        Err(KernelError::AmbiguousLiSort { .. }) => {
            sort_poynting(q, psi, direction, a3n, a6n, layer_index)
        }
        Err(e) => Err(e),
    }
}

/// Recover the longitudinal `E_z, H_z` components per PP2017
/// Eqs. (17) and (18). Used by the Poynting-vector sort and by
/// the field reconstruction in [`crate::kernel::field`].
pub fn longitudinal_components(
    psi_column: &Vector4<C64>,
    a3n: &[C64; 6],
    a6n: &[C64; 6],
) -> KernelResult<(C64, C64)> {
    let ex = psi_column[0];
    let hy = psi_column[1];
    let ey = psi_column[2];
    let neg_hx = psi_column[3];
    let hx = -neg_hx;
    let ez = a3n[0] * ex + a3n[1] * ey + a3n[3] * hx + a3n[4] * hy;
    let hz = a6n[0] * ex + a6n[1] * ey + a6n[3] * hx + a6n[4] * hy;
    Ok((ez, hz))
}
