//! Phantom-state Pipeline composing the six pipeline stages.

use crate::error::{KernelError, KernelResult};
use crate::kernel::coefficients;
use crate::kernel::constitutive::{self, MMatrix};
use crate::kernel::delta::{self, DeltaMatrix};
use crate::kernel::field;
use crate::kernel::geometry;
use crate::kernel::interface::{self, Gamma};
use crate::kernel::modes::{self, LayerModes};
use crate::kernel::propagate;
use crate::material::rotation::rotate_principal;
use crate::stack::Stack;
use crate::types::scalar::{wavelength_m_from_omega, C64};
use crate::types::tensor::LabTensor;
use nalgebra::Matrix4;
use std::marker::PhantomData;

pub use crate::kernel::coefficients::Amplitudes;
pub use crate::kernel::field::{FieldProfile, Polarization};

#[derive(Debug)]
pub struct Pipeline<S> {
    stack: Stack,
    omega_rad_per_s: f64,
    theta_rad: f64,
    state: S,
    _marker: PhantomData<S>,
}

#[derive(Debug, Default)]
pub struct Init;

#[derive(Debug)]
pub struct Constituted {
    pub m_matrices: Vec<MMatrix>,
    pub xi: C64,
}

#[derive(Debug)]
pub struct Reduced {
    pub delta_matrices: Vec<DeltaMatrix>,
    pub a3n: Vec<[C64; 6]>,
    pub a6n: Vec<[C64; 6]>,
    pub xi: C64,
}

#[derive(Debug)]
pub struct Eigenmoded {
    pub modes: Vec<LayerModes>,
    pub xi: C64,
}

#[derive(Debug)]
pub struct Interfaced {
    pub a_incident: Matrix4<C64>,
    pub a_layers: Vec<Matrix4<C64>>,
    pub a_substrate: Matrix4<C64>,
    pub hat_gamma_layers: Vec<[Gamma; 4]>,
    pub layer_modes: Vec<LayerModes>,
    pub xi: C64,
}

#[derive(Debug)]
pub struct Propagated {
    pub gamma_passler: Matrix4<C64>,
    pub gamma_yeh: Matrix4<C64>,
    pub interfaced: Interfaced,
}

#[derive(Debug)]
pub struct Solved {
    pub amplitudes: Amplitudes,
    pub propagated: Propagated,
}

fn layer_tensors(layer: &crate::stack::Layer) -> KernelResult<(LabTensor, LabTensor)> {
    let eps_diag = layer
        .material
        .epsilon_principal
        .to_epsilon_principal_diag()?;
    eps_diag.validate_passive()?;
    let eps_lab = rotate_principal(eps_diag, layer.material.euler_zxz_rad)?;
    let mu_lab = if let Some(mu) = layer.material.mu_principal {
        let d = mu.to_epsilon_principal_diag()?;
        d.validate_passive()?;
        rotate_principal(d, layer.material.euler_zxz_rad)?
    } else {
        LabTensor::identity()
    };
    Ok((eps_lab, mu_lab))
}

fn scalar_mu(mu_lab: &LabTensor) -> C64 {
    let m = mu_lab.as_matrix();
    (m[(0, 0)] + m[(1, 1)] + m[(2, 2)]) / 3.0_f64
}

fn incident_epsilon_scalar(stack: &Stack) -> KernelResult<C64> {
    let (eps_lab, _) = layer_tensors(&stack.incident)?;
    let m = eps_lab.as_matrix();
    Ok((m[(0, 0)] + m[(1, 1)] + m[(2, 2)]) / 3.0_f64)
}

impl Pipeline<Init> {
    pub fn new(stack: Stack, omega_rad_per_s: f64, theta_rad: f64) -> Self {
        Self {
            stack,
            omega_rad_per_s,
            theta_rad,
            state: Init,
            _marker: PhantomData,
        }
    }

    pub fn build_constitutive(self) -> KernelResult<Pipeline<Constituted>> {
        let _ = wavelength_m_from_omega(self.omega_rad_per_s);
        let eps_inc = incident_epsilon_scalar(&self.stack)?;
        let xi = geometry::tangential_xi(eps_inc, self.theta_rad)?;
        let mut m_matrices = Vec::with_capacity(self.stack.layers.len() + 2);
        for layer in std::iter::once(&self.stack.incident)
            .chain(self.stack.layers.iter())
            .chain(std::iter::once(&self.stack.substrate))
        {
            let (eps, mu) = layer_tensors(layer)?;
            m_matrices.push(constitutive::build_m(eps, mu)?);
        }
        Ok(Pipeline {
            stack: self.stack,
            omega_rad_per_s: self.omega_rad_per_s,
            theta_rad: self.theta_rad,
            state: Constituted { m_matrices, xi },
            _marker: PhantomData,
        })
    }
}

impl Pipeline<Constituted> {
    pub fn reduce_to_delta(self) -> KernelResult<Pipeline<Reduced>> {
        let xi = self.state.xi;
        let mut deltas = Vec::with_capacity(self.state.m_matrices.len());
        let mut a3v = Vec::with_capacity(self.state.m_matrices.len());
        let mut a6v = Vec::with_capacity(self.state.m_matrices.len());
        for m in &self.state.m_matrices {
            let b = delta::compute_b(m)?;
            a3v.push(delta::compute_a3n(m, b, xi)?);
            a6v.push(delta::compute_a6n(m, b, xi)?);
            deltas.push(delta::build_delta(m, xi)?);
        }
        Ok(Pipeline {
            stack: self.stack,
            omega_rad_per_s: self.omega_rad_per_s,
            theta_rad: self.theta_rad,
            state: Reduced {
                delta_matrices: deltas,
                a3n: a3v,
                a6n: a6v,
                xi,
            },
            _marker: PhantomData,
        })
    }
}

impl Pipeline<Reduced> {
    pub fn solve_eigenmodes(self) -> KernelResult<Pipeline<Eigenmoded>> {
        let xi = self.state.xi;
        let mut modes = Vec::with_capacity(self.state.delta_matrices.len());
        for (k, d) in self.state.delta_matrices.iter().enumerate() {
            let (q, psi) = modes::solve_eigenmodes(d)?;
            let dir = modes::partition_modes(&q)?;
            let lm = modes::sort_modes_with_fallback(
                &q,
                &psi,
                &dir,
                &self.state.a3n[k],
                &self.state.a6n[k],
                k,
            )?;
            modes.push(lm);
        }
        Ok(Pipeline {
            stack: self.stack,
            omega_rad_per_s: self.omega_rad_per_s,
            theta_rad: self.theta_rad,
            state: Eigenmoded { modes, xi },
            _marker: PhantomData,
        })
    }
}

impl Pipeline<Eigenmoded> {
    pub fn build_interfaces(self) -> KernelResult<Pipeline<Interfaced>> {
        let xi = self.state.xi;
        let n = self.stack.layers.len();
        let mut a_inc = Matrix4::zeros();
        let mut a_layers = Vec::with_capacity(n);
        let mut a_sub = Matrix4::zeros();
        let mut hats = Vec::with_capacity(n + 2);
        let mut idx = 0;
        for layer in std::iter::once(&self.stack.incident)
            .chain(self.stack.layers.iter())
            .chain(std::iter::once(&self.stack.substrate))
        {
            let (eps, mu) = layer_tensors(layer)?;
            let mu_s = scalar_mu(&mu);
            let modes_l = &self.state.modes[idx];
            let hat = interface::hat_gamma_quadruplet(&eps, mu_s, xi, modes_l)?;
            let a = interface::build_a(&hat, modes_l, mu_s, xi)?;
            hats.push(hat);
            if idx == 0 {
                a_inc = a;
            } else if idx <= n {
                a_layers.push(a);
            } else {
                a_sub = a;
            }
            idx += 1;
        }
        Ok(Pipeline {
            stack: self.stack,
            omega_rad_per_s: self.omega_rad_per_s,
            theta_rad: self.theta_rad,
            state: Interfaced {
                a_incident: a_inc,
                a_layers,
                a_substrate: a_sub,
                hat_gamma_layers: hats,
                layer_modes: self.state.modes.clone(),
                xi,
            },
            _marker: PhantomData,
        })
    }
}

impl Pipeline<Interfaced> {
    pub fn propagate(self) -> KernelResult<Pipeline<Propagated>> {
        let n = self.stack.layers.len();
        let mut triples = Vec::with_capacity(n);
        for i in 0..n {
            let a = &self.state.a_layers[i];
            let ai = a.try_inverse().ok_or_else(|| {
                KernelError::SingularCoefficientDenominator(a[(0, 0)].re.abs())
            })?;
            let lm = &self.state.layer_modes[i + 1];
            let th = self.stack.layers[i].thickness_nm;
            let p = propagate::build_p(&lm.q, th, self.omega_rad_per_s)?;
            triples.push((*a, p, ai));
        }
        let gamma_passler = propagate::assemble_gamma(
            &self.state.a_incident,
            &triples,
            &self.state.a_substrate,
        )?;
        let gamma_yeh = propagate::permute_to_yeh(&gamma_passler);
        let interfaced = Interfaced {
            a_incident: self.state.a_incident,
            a_layers: self.state.a_layers,
            a_substrate: self.state.a_substrate,
            hat_gamma_layers: self.state.hat_gamma_layers,
            layer_modes: self.state.layer_modes,
            xi: self.state.xi,
        };
        Ok(Pipeline {
            stack: self.stack,
            omega_rad_per_s: self.omega_rad_per_s,
            theta_rad: self.theta_rad,
            state: Propagated {
                gamma_passler,
                gamma_yeh,
                interfaced,
            },
            _marker: PhantomData,
        })
    }
}

impl Pipeline<Propagated> {
    pub fn solve_amplitudes(self) -> KernelResult<Pipeline<Solved>> {
        let amp = coefficients::build_amplitudes(&self.state.gamma_yeh)?;
        Ok(Pipeline {
            stack: self.stack,
            omega_rad_per_s: self.omega_rad_per_s,
            theta_rad: self.theta_rad,
            state: Solved {
                amplitudes: amp,
                propagated: Propagated {
                    gamma_passler: self.state.gamma_passler,
                    gamma_yeh: self.state.gamma_yeh,
                    interfaced: Interfaced {
                        a_incident: self.state.interfaced.a_incident,
                        a_layers: self.state.interfaced.a_layers.clone(),
                        a_substrate: self.state.interfaced.a_substrate,
                        hat_gamma_layers: self.state.interfaced.hat_gamma_layers.clone(),
                        layer_modes: self.state.interfaced.layer_modes.clone(),
                        xi: self.state.interfaced.xi,
                    },
                },
            },
            _marker: PhantomData,
        })
    }
}

impl Pipeline<Solved> {
    pub fn amplitudes(&self) -> &Amplitudes {
        &self.state.amplitudes
    }

    pub fn reconstruct_field_at(
        &self,
        layer_index: usize,
        z_nm: f64,
        polarization: Polarization,
    ) -> KernelResult<FieldProfile> {
        let n = self.stack.layers.len();
        if layer_index >= n {
            return Err(KernelError::InvalidGeometry(
                "layer_index out of range".into(),
            ));
        }
        let amp = &self.state.amplitudes;
        let v_sub = field::substrate_mode_vector(amp, polarization);
        let mut v = v_sub;
        let i = &self.state.propagated.interfaced;
        let mut j = n + 1;
        while j > layer_index + 1 {
            let l_j = if j == n + 1 {
                interface::build_l(&i.a_layers[n - 1], &i.a_substrate)?
            } else {
                interface::build_l(&i.a_layers[j - 2], &i.a_layers[j - 1])?
            };
            v = l_j * v;
            if j <= n {
                let p = propagate::build_p(
                    &i.layer_modes[j - 1].q,
                    self.stack.layers[j - 1].thickness_nm,
                    self.omega_rad_per_s,
                )?;
                v = p * v;
            }
            j -= 1;
        }
        let lm = &i.layer_modes[layer_index + 1];
        let hat = i.hat_gamma_layers[layer_index + 1];
        field::reconstruct(z_nm, &v, lm, &hat, self.omega_rad_per_s)
    }
}

pub fn compute_amplitudes(
    stack: &Stack,
    omega_rad_per_s: f64,
    theta_rad: f64,
) -> KernelResult<Amplitudes> {
    stack.validate()?;
    let p = Pipeline::new(stack.clone(), omega_rad_per_s, theta_rad)
        .build_constitutive()?
        .reduce_to_delta()?
        .solve_eigenmodes()?
        .build_interfaces()?
        .propagate()?
        .solve_amplitudes()?;
    Ok(p.state.amplitudes)
}

pub fn compute_field(
    stack: &Stack,
    omega_rad_per_s: f64,
    theta_rad: f64,
    layer_index: usize,
    z_nm: f64,
    polarization: Polarization,
) -> KernelResult<FieldProfile> {
    stack.validate()?;
    let p = Pipeline::new(stack.clone(), omega_rad_per_s, theta_rad)
        .build_constitutive()?
        .reduce_to_delta()?
        .solve_eigenmodes()?
        .build_interfaces()?
        .propagate()?
        .solve_amplitudes()?;
    p.reconstruct_field_at(layer_index, z_nm, polarization)
}
