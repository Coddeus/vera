use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

/// Vertex for Vera
#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct Veratex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
    #[format(R32_UINT)]
    entity_id: u32,
}

impl Veratex {
    pub fn new(x: f32, y: f32, entity_id: u32) -> Self {
        Veratex { position: [x, y], entity_id: entity_id }
    }
}


pub trait Vertices {
    fn vertices(self) -> [Veratex ; 3];
}