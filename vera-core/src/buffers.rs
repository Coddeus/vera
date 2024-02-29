use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex, padded::Padded};

/// A vertex with expected position and color, given as input to the graphics pipeline.
#[derive(BufferContents, Vertex, Debug, Clone, Copy)]
#[repr(C)]
pub(crate) struct BaseVertex {
    /// The (x, y) [normalized-square-centered](broken_link) coordinates of this vertex.
    #[format(R32G32B32A32_SFLOAT)]
    pub(crate) position: [f32; 4],
    /// The rgba color of this vertex.
    #[format(R32G32B32A32_SFLOAT)]
    pub(crate) color: [f32; 4],
    /// The coordinates of the texture on this vertex.
    #[format(R32G32_SFLOAT)]
    pub(crate) tex_coord: [f32; 2],
    /// The id of the texture drawn on this vertex.
    #[format(R32_UINT)]
    pub(crate) tex_id: u32,
    /// The id of the entity this vertex belongs to.
    #[format(R32_UINT)]
    pub(crate) entity_id: u32,
}
impl Default for BaseVertex {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0, 1.0],
            color: [0.5, 1.0, 0.8, 1.0],
            tex_coord: [0.0, 0.0],
            tex_id: 0,
            entity_id: 0,
        }
    }
}

/// Matrix transformation data.
#[derive(BufferContents, Debug, Clone, Copy)]
#[repr(C)]
pub(crate) struct MatrixT {
    /// The value of the transformation
    pub(crate) mat: [f32; 16],
}

/// Vector transformation data.
#[derive(BufferContents, Debug)]
#[repr(C)]
pub(crate) struct VectorT {
    /// The value of the transformation
    pub(crate) vec: [f32; 4],
}

/// Per-entity data.
#[derive(BufferContents, Debug)]
#[repr(C)]
pub(crate) struct Entity {
    pub(crate) parent_id: Padded<u32, 12>,
}

/// A matrix transformation
#[derive(BufferContents, Debug)]
#[repr(C)]
pub(crate) struct MatrixTransformation {
    /// The kind of transformation
    pub(crate) val: [f32; 3],
    pub(crate) ty: u32, 
    pub(crate) start: f32,
    pub(crate) end: f32,
    pub(crate) evolution: Padded<u32, 4>,
}
impl Default for MatrixTransformation {
    fn default() -> Self {
        Self {
            ty: 0,
            val: [0.0, 0.0, 0.0],
            start: 0.0,
            end: 0.0,
            evolution: Padded(0),
        }
    }
}

/// A color transformation
#[derive(BufferContents, Debug)]
#[repr(C)]
pub(crate) struct ColorTransformation {
    pub(crate) val: [f32; 4],
    pub(crate) ty: u32,
    pub(crate) start: f32,
    pub(crate) end: f32,
    pub(crate) evolution: u32,
}
impl Default for ColorTransformation {
    fn default() -> Self {
        Self {
            val: [0.0, 0.0, 0.0, 0.0],
            ty: 0,
            start: 0.0,
            end: 0.0,
            evolution: 0,
        }
    }
}

/// A matrix transformer
#[derive(BufferContents, Debug, Clone, Copy)]
#[repr(C)]
pub(crate) struct MatrixTransformer {
    pub(crate) mat: [f32; 16],
    pub(crate) range: Padded<[u32; 2], 8>,
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
            range: Padded([offset, offset+length]),
        }
    }
}

/// A color transformer
#[derive(BufferContents, Debug, Clone, Copy)]
#[repr(C)]
pub(crate) struct ColorTransformer {
    pub(crate) vec: [f32; 4],
    pub(crate) range: Padded<[u32; 2], 8>,
}
impl ColorTransformer {
    pub(crate) fn from_loc(length: u32, offset: u32, col: [f32; 4]) -> Self {
        Self {
            vec: col,
            range: Padded([offset, offset+length]),
        }
    }
}

/// General-purpose uniform data used in the compute shader.
/// Used as push constant
#[derive(Debug, Clone, BufferContents)]
#[repr(C)]
pub(crate) struct CSGeneral {
    /// The time elapsed since the beginning.
    pub(crate) time: f32,
}

/// General-purpose uniform data used in the vertex shader.
/// Used as push constant
#[derive(Debug, Clone, BufferContents)]
#[repr(C)]
pub(crate) struct VSGeneral {
    /// The vpr matrix (view, projection, resolution)
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