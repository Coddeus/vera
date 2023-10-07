use vulkano::buffer::BufferContents;

#[derive(Debug, Clone, BufferContents)]
#[repr(C)]
pub struct UniformData {
    model_matrix: [f32 ; 9],
}

impl UniformData {
    pub fn empty() -> Self {
        UniformData { 
            model_matrix: [
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
            ]
        }
    }
}