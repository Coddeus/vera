use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

/// A base vertex for Vera, meant to be given as input to the graphics pipeline. Use this to create custom shapes.
#[derive(BufferContents, Vertex, Debug)]
#[repr(C)]
pub struct Veratex {
    #[format(R32G32_SFLOAT)]
    pub position: [f32; 2],
    #[format(R32_UINT)]
    pub entity_id: usize,
}
impl Veratex {
    pub fn new(x: f32, y: f32, entity_id: usize) -> Self {
        Veratex { position: [x, y], entity_id }
    }
}

/// Any shape. This is what `fn new()` of specific shapes return.
/// - `vertices` define the shape, each groups of three `Veratex` forming a triangle.
pub struct Shape {
    pub vertices: Vec<Veratex>,
}
impl Shape {

}

pub struct Triangle;

impl Triangle {
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32, x3: f32, y3: f32) -> Shape {
        Shape {
            vertices: vec![
                Veratex::new(x1, y1, 0),
                Veratex::new(x2, y2, 0),
                Veratex::new(x3, y3, 0),
            ],
        }
    }
}