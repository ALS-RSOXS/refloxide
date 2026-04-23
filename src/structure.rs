/// Core structure data types for the simulation.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Medium {
    refractive_index: f64,
}

impl Medium {
    /// Creates a semi-infinite optical medium.
    pub fn new(refractive_index: f64) -> Result<Self, StructureError> {
        if !refractive_index.is_finite() || refractive_index <= 0.0 {
            return Err(StructureError::InvalidRefractiveIndex(refractive_index));
        }
        Ok(Self { refractive_index })
    }

    /// Returns the refractive index of this medium.
    pub fn refractive_index(self) -> f64 {
        self.refractive_index
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FilmLayer {
    thickness: f64,
    refractive_index: f64,
}

impl FilmLayer {
    /// Creates a finite-thickness layer in the stratified stack.
    pub fn new(thickness: f64, refractive_index: f64) -> Result<Self, StructureError> {
        if !thickness.is_finite() || thickness < 0.0 {
            return Err(StructureError::InvalidThickness(thickness));
        }
        if !refractive_index.is_finite() || refractive_index <= 0.0 {
            return Err(StructureError::InvalidRefractiveIndex(refractive_index));
        }
        Ok(Self {
            thickness,
            refractive_index,
        })
    }

    /// Returns the layer thickness.
    pub fn thickness(self) -> f64 {
        self.thickness
    }

    /// Returns the refractive index of the layer.
    pub fn refractive_index(self) -> f64 {
        self.refractive_index
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Structure {
    fronting: Medium,
    backing: Medium,
    layers: Vec<FilmLayer>,
    interface_roughness: Vec<f64>,
}

impl Structure {
    /// Creates a stratified structure with explicit fronting and backing media.
    ///
    /// `interface_roughness` must have `layers.len() + 1` values:
    /// fronting->layer0, layer0->layer1, ..., layerN-1->backing.
    pub fn new(
        fronting: Medium,
        layers: Vec<FilmLayer>,
        backing: Medium,
        interface_roughness: Vec<f64>,
    ) -> Result<Self, StructureError> {
        let expected_interfaces = layers.len() + 1;
        if interface_roughness.len() != expected_interfaces {
            return Err(StructureError::InvalidInterfaceCount {
                expected: expected_interfaces,
                got: interface_roughness.len(),
            });
        }

        for roughness in &interface_roughness {
            if !roughness.is_finite() || *roughness < 0.0 {
                return Err(StructureError::InvalidRoughness(*roughness));
            }
        }

        Ok(Self {
            fronting,
            backing,
            layers,
            interface_roughness,
        })
    }

    /// Creates a structure where all interfaces share the same roughness.
    pub fn with_uniform_roughness(
        fronting: Medium,
        layers: Vec<FilmLayer>,
        backing: Medium,
        roughness: f64,
    ) -> Result<Self, StructureError> {
        if !roughness.is_finite() || roughness < 0.0 {
            return Err(StructureError::InvalidRoughness(roughness));
        }
        let interface_count = layers.len() + 1;
        Self::new(fronting, layers, backing, vec![roughness; interface_count])
    }

    /// Returns the fronting medium.
    pub fn fronting(&self) -> Medium {
        self.fronting
    }

    /// Returns the backing medium.
    pub fn backing(&self) -> Medium {
        self.backing
    }

    /// Returns the finite-thickness interior layers.
    pub fn layers(&self) -> &[FilmLayer] {
        &self.layers
    }

    /// Returns interface roughness values from top to bottom.
    pub fn interface_roughness(&self) -> &[f64] {
        &self.interface_roughness
    }

    /// Returns the number of interior finite-thickness layers.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum StructureError {
    InvalidThickness(f64),
    InvalidRoughness(f64),
    InvalidRefractiveIndex(f64),
    InvalidInterfaceCount { expected: usize, got: usize },
}

impl core::fmt::Display for StructureError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidThickness(v) => {
                write!(f, "thickness must be finite and >= 0, got {v}")
            }
            Self::InvalidRoughness(v) => {
                write!(f, "roughness must be finite and >= 0, got {v}")
            }
            Self::InvalidRefractiveIndex(v) => {
                write!(f, "refractive index must be finite and > 0, got {v}")
            }
            Self::InvalidInterfaceCount { expected, got } => {
                write!(
                    f,
                    "invalid interface roughness count: expected {expected}, got {got}"
                )
            }
        }
    }
}

impl std::error::Error for StructureError {}
