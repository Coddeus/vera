use vulkano::buffer::BufferContents;

/// Uniform data for one entity
#[derive(Debug, Clone, BufferContents)]
#[repr(C)]
pub(crate) struct UniformData {
    /// The transformation matrix of an element (enables translation, rotation, scaling)
    /// ```
    /// RotateCos * ScaleX  , -RotateSin        , TranslateX, alignment(unused),
    /// RotateSin           , RotateCos * ScaleY, TranslateY, alignment(unused),
    /// 0.0                 , 0.0               , 1.0       , alignment(unused),
    /// ```
    model_matrix: [f32 ; 12],

    // color: [f32 ; 4],
}

impl UniformData {
    /// Returns an identity matrix, applying no transformation.
    pub(crate) fn empty() -> Self {
        UniformData { 
            model_matrix: [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ]
        }
    }
}