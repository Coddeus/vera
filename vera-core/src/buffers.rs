use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

/// An already-transformed vertex with a position and a color, given as input to the graphics pipeline.
#[derive(BufferContents, Vertex, Debug)]
#[repr(C)]
pub(crate) struct VSInput {
    /// The (x, y) [normalized-square-centered](broken_link) coordinates of this vertex.
    #[format(R32G32B32_SFLOAT)]
    pub(crate) position: [f32; 3],
    /// The rgba color of this vertex.
    #[format(R32G32B32A32_SFLOAT)]
    pub(crate) color: [f32; 4],
}
impl VSInput {
    pub(crate) fn new(position: [f32; 3], color: [f32; 4]) -> Self {
        VSInput {
            position,
            color, 
        }
    }
}

/// The original, unmodified vertex data, set once for the descriptor set to read.
#[derive(BufferContents, Debug)]
#[repr(C)]
pub(crate) struct BaseVertex {
    /// The (x, y) [normalized-square-centered](broken_link) coordinates of this vertex.
    pub(crate) position: [f32; 3],
    /// The rgba color of this vertex.
    pub(crate) color: [f32; 4],
    /// The id of this vertex.
    pub(crate) vertex_id: u32,
}
impl BaseVertex {
    pub(crate) fn new(position: [f32; 3], color: [f32; 4], vertex_id: u32) -> Self {
        BaseVertex {
            position,
            color,
            vertex_id,
        }
    }
}

/// The data read and updated via the compute shader that recreates the vertex buffer every frame.
#[derive(BufferContents, Debug)]
#[repr(C)]
pub(crate) struct MatrixTransformers {
    
}
impl MatrixTransformers {
    pub(crate) fn new(position: [f32; 3], color: [f32; 4]) -> Self {
        MatrixTransformers {
            position,
            color, 
        }
    }
}

/// Additional vertex data for one vertex
#[derive(Debug, Clone, BufferContents)]
#[repr(C)]
pub(crate) struct TransformVertex {
}
impl TransformVertex {
    /// Returns an identity matrix, applying no transformation.
    pub(crate) fn new() -> Self {
        TransformVertex { 
            vertex_matrix0: [1.0, 0.0, 0.0, 0.0],
            vertex_matrix1: [0.0, 1.0, 0.0, 0.0],
            vertex_matrix2: [0.0, 0.0, 1.0, 0.0],
            vertex_matrix3: [0.0, 0.0, 0.0, 1.0],
        }
    }
}

/// The data read and updated via the compute shader that recreates the vertex buffer every frame.
#[derive(BufferContents, Debug)]
#[repr(C)]
pub(crate) struct VertexMatrix {
    /// The first vec4 of the vertex matrix
    pub(crate) vertex_matrix0: [f32 ; 4],
    /// The second vec4 of the vertex matrix
    pub(crate) vertex_matrix1: [f32 ; 4],
    /// The third vec4 of the vertex matrix
    pub(crate) vertex_matrix2: [f32 ; 4],
    /// The fourth vec4 of the vertex matrix
    pub(crate) vertex_matrix3: [f32 ; 4],
}
impl CSInput {
    pub(crate) fn new(position: [f32; 3], color: [f32; 4]) -> Self {
        CSInput {
            position,
            color, 
        }
    }
}

/// Additional vertex data for one vertex
#[derive(Debug, Clone, BufferContents)]
#[repr(C)]
pub(crate) struct TransformVertex {
}
impl TransformVertex {
    /// Returns an identity matrix, applying no transformation.
    pub(crate) fn new() -> Self {
        TransformVertex { 
            vertex_matrix0: [1.0, 0.0, 0.0, 0.0],
            vertex_matrix1: [0.0, 1.0, 0.0, 0.0],
            vertex_matrix2: [0.0, 0.0, 1.0, 0.0],
            vertex_matrix3: [0.0, 0.0, 0.0, 1.0],
        }
    }
}

/// General-purpose uniform data
#[derive(Debug, Clone, BufferContents)]
#[repr(C)]
pub(crate) struct GeneralData {
    /// The projection matrix
    pub(crate) projection_matrix: [f32 ; 16],
    /// The view matrix
    pub(crate) view_matrix: [f32 ; 16],
    /// The viewport resolution. Used to unstretch the shapes within the viewport.
    pub(crate) resolution: [f32 ; 2],
    /// The time elapsed since the beginning.
    pub(crate) time: f32,
}
impl GeneralData {
    // /// Returns data applying no transformation.
    // pub(crate) fn new() -> Self {
    //     GeneralData {
    //         projection_matrix: [
    //             1.0, 0.0, 0.0, 0.0,
    //             0.0, 1.0, 0.0, 0.0,
    //             0.0, 0.0, 1.0, 0.0,
    //             0.0, 0.0, 0.0, 1.0,
    //         ],
    //         view_matrix: [
    //             1.0, 0.0, 0.0, 0.0,
    //             0.0, 1.0, 0.0, 0.0,
    //             0.0, 0.0, 1.0, 0.0,
    //             0.0, 0.0, 0.0, 1.0,
    //         ],
    //         resolution: [100.0, 100.0],
    //         time: 0.0,
    //     }
    // }
    /// Returns uniform data applying no other transformation than setting the resolution to the given value.
    pub(crate) fn from_resolution(inner_size: [f32 ; 2]) -> Self {
        GeneralData {
            projection_matrix: [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
            view_matrix: [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
            resolution: inner_size,
            time: 0.0,
        }
    }
    // /// Sets the resolution to the given window size
    // pub(crate) fn set_resolution(&mut self, inner_size: [f32 ; 2]) {
    //     self.resolution = inner_size;
    // }
}
impl Default for GeneralData {
    fn default() -> Self {
        GeneralData {
            projection_matrix: [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
            view_matrix: [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
            resolution: [1.0, 1.0],
            time: 0.0,
        }
    }
}

/// Uniform data for one entity
#[derive(Debug, Clone, BufferContents)]
#[repr(C)]
pub(crate) struct EntityData {
    /// The model matrix
    pub(crate) model_matrix: [f32 ; 16],
}
impl EntityData {
    /// Returns an identity matrix, applying no transformation.
    pub(crate) fn new() -> Self {
        EntityData {
            model_matrix: [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        }
    }
}