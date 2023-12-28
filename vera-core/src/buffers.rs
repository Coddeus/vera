use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

/// An already-transformed vertex with a position and a color, given as input to the graphics pipeline.
#[derive(BufferContents, Vertex, Debug)]
#[repr(C)]
pub(crate) struct VSInput {
    /// The (x, y) [normalized-square-centered](broken_link) coordinates of this vertex.
    #[format(R32G32B32_SFLOAT)]
    pub(crate) position: [f32; 4],
    /// The rgba color of this vertex.
    #[format(R32G32B32A32_SFLOAT)]
    pub(crate) color: [f32; 4],
}
impl VSInput {
    pub(crate) fn new(position: [f32; 4], color: [f32; 4]) -> Self {
        Self {
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
    pub(crate) position: [f32; 4],
    /// The rgba color of this vertex.
    pub(crate) color: [f32; 4],
    /// The id of this vertex.
    pub(crate) entity_id: u32,
}
impl BaseVertex {
    pub(crate) fn new(position: [f32; 4], color: [f32; 4], entity_id: u32) -> Self {
        Self {
            position,
            color,
            entity_id,
        }
    }
}

/// The data read and updated via the compute shader that recreates the vertex buffer every frame.
#[derive(BufferContents, Debug)]
#[repr(C)]
pub(crate) struct Entity {
    entity_id: u32,
    parent_id: u32,
}
impl Entity {
    pub(crate) fn new(entity_id: u32, parent_id: u32) -> Self {
        Self {
            entity_id,
            parent_id,
        }
    }
}

/// A matrix transformation
#[derive(BufferContents, Debug)]
#[repr(C)]
pub(crate) struct MatrixTransformation {
    ty: u32, 
    val: [f32; 9], 
    start: f32, 
    end: f32, 
    evolution: u32,
}
impl MatrixTransformation {
    pub(crate) fn new(ty: u32, val: [f32; 9], start: f32, end: f32, evolution: u32) -> Self {
        Self {
            ty,
            val,
            start,
            end,
            evolution,
        }
    }
}

/// A color transformation
#[derive(BufferContents, Debug)]
#[repr(C)]
pub(crate) struct ColorTransformation {
    val: [f32; 9],
    start: f32,
    end: f32,
    ty: u32,
    evolution: u32,
}
impl ColorTransformation {
    pub(crate) fn new(ty: u32, val: [f32; 9], start: f32, end: f32, evolution: u32) -> Self {
        Self {
            ty,
            val,
            start,
            end,
            evolution,
        }
    }
}

/// A matrix transformer
#[derive(BufferContents, Debug)]
#[repr(C)]
pub(crate) struct MatrixTransformer {
    mat: [f32; 16],
    range: [u32; 2],
}
impl MatrixTransformer {
    pub(crate) fn new(first: u32, lastplusone: u32) -> Self {
        Self {
            mat: [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
            range: [first, lastplusone],
        }
    }
}

/// A color transformer
#[derive(BufferContents, Debug)]
#[repr(C)]
pub(crate) struct ColorTransformer {
    vec: [f32; 4],
    range: [u32; 2],
}
impl ColorTransformer {
    pub(crate) fn new(first: u32, lastplusone: u32) -> Self {
        Self {
            vec: [1.0, 1.0, 1.0, 1.0], 
            range: [first, lastplusone],
        }
    }
}

/// General-purpose uniform data
#[derive(Debug, Clone, BufferContents)]
#[repr(C)]
pub(crate) struct GeneralData {
    /// The vpr matrix
    pub(crate) mat: [f32 ; 16],
    /// The time elapsed since the beginning.
    pub(crate) time: f32,
}
impl GeneralData {
    /// Returns uniform data applying no other transformation than setting the resolution to the given value.
    pub(crate) fn new(mat: [f32; 16], time: f32) -> Self {
        Self {
            mat,
            time,
        }
    }
    // /// Sets the resolution to the given window size
    // pub(crate) fn set_resolution(&mut self, inner_size: [f32 ; 2]) {
    //     self.resolution = inner_size;
    // }
}
impl Default for GeneralData {
    fn default() -> Self {
        Self {
            mat: [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
            time: 0.0,
        }
    }
}