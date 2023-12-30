use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

/// A vertex with expected position and color, given as input to the graphics pipeline.
#[derive(BufferContents, Vertex, Debug)]
#[repr(C)]
pub(crate) struct BaseVertex {
    /// The (x, y) [normalized-square-centered](broken_link) coordinates of this vertex.
    #[format(R32G32B32_SFLOAT)]
    pub(crate) position: [f32; 4],
    /// The rgba color of this vertex.
    #[format(R32G32B32A32_SFLOAT)]
    pub(crate) color: [f32; 4],
    /// The id of this vertex.
    #[format(R32_UINT)]
    pub(crate) entity_id: u32,
}
impl Default for BaseVertex {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0, 1.0],
            color: [0.5, 1.0, 0.8, 1.0],
            entity_id: 0,
        }
    }
}

/// The original, unmodified vertex data, set once for the descriptor set to read.
#[derive(BufferContents, Vertex, Debug)]
#[repr(C)]
pub(crate) struct MatrixT {
    /// The (x, y) [normalized-square-centered](broken_link) coordinates of this vertex.
    #[name("transform")]
    #[format(R32G32B32_SFLOAT)]
    pub(crate) mat: [f32; 16],
}

/// The original, unmodified vertex data, set once for the descriptor set to read.
#[derive(BufferContents, Vertex, Debug)]
#[repr(C)]
pub(crate) struct VectorT {
    /// The (x, y) [normalized-square-centered](broken_link) coordinates of this vertex.
    #[name("color")]
    #[format(R32G32B32A32_SFLOAT)]
    pub(crate) vec: [f32; 4],
}

/// The data read and updated via the compute shader that recreates the vertex buffer every frame.
#[derive(BufferContents, Debug)]
#[repr(C)]
pub(crate) struct Entity {
    pub(crate) parent_id: u32,
}

/// A matrix transformation
#[derive(BufferContents, Debug)]
#[repr(C)]
pub(crate) struct MatrixTransformation {
    /// The kind of transformation
    pub(crate) ty: u32, 
    pub(crate) val: [f32; 3],
    pub(crate) start: f32,
    pub(crate) end: f32,
    pub(crate) evolution: u32,
}

/// A color transformation
#[derive(BufferContents, Debug)]
#[repr(C)]
pub(crate) struct ColorTransformation {
    pub(crate) ty: u32,
    pub(crate) val: [f32; 4],
    pub(crate) start: f32,
    pub(crate) end: f32,
    pub(crate) evolution: u32,
}

/// A matrix transformer
#[derive(BufferContents, Debug)]
#[repr(C)]
pub(crate) struct MatrixTransformer {
    pub(crate) mat: [f32; 16],
    pub(crate) range: [u32; 2],
}
impl MatrixTransformer {
    pub(crate) fn from_lo(length: u32, offset: u32) -> Self {
        Self {
            mat: [
                1.0, 0.0, 0.0, 0.0, 
                0.0, 1.0, 0.0, 0.0, 
                0.0, 0.0, 1.0, 0.0, 
                0.0, 0.0, 0.0, 1.0, 
            ],
            range: [offset, offset+length],
        }
    }
}

/// A color transformer
#[derive(BufferContents, Debug)]
#[repr(C)]
pub(crate) struct ColorTransformer {
    pub(crate) vec: [f32; 4],
    pub(crate) range: [u32; 2],
}
impl ColorTransformer {
    pub(crate) fn from_lo(length: u32, offset: u32) -> Self {
        Self {
            vec: [0.0, 0.0, 0.0, 1.0],
            range: [offset, offset+length],
        }
    }
}

/// General-purpose uniform data used in the compute shader.
#[derive(Debug, Clone, BufferContents)]
#[repr(C)]
pub(crate) struct CSGeneral {
    /// The vpr matrix
    pub(crate) entity_count: u64,
    /// The time elapsed since the beginning.
    pub(crate) time: f32,
}

/// General-purpose uniform data used in the vertex shader.
#[derive(Debug, Clone, BufferContents)]
#[repr(C)]
pub(crate) struct VSGeneral {
    /// The vpr matrix
    pub(crate) mat: [f32 ; 16],
}
// impl Default for GeneralData {
//     fn default() -> Self {
//         Self {
//             mat: [
//                 1.0, 0.0, 0.0, 0.0,
//                 0.0, 1.0, 0.0, 0.0,
//                 0.0, 0.0, 1.0, 0.0,
//                 0.0, 0.0, 0.0, 1.0,
//             ],
//             time: 0.0,
//         }
//     }
// }