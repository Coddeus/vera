use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

/// A base vertex for Vera, meant to be given as input to the graphics pipeline.
#[derive(BufferContents, Vertex, Debug)]
#[repr(C)]
pub struct Veratex {
    /// The (x, y) [normalized-square-centered](broken link) coordinates of the vertex.
    #[format(R32G32_SFLOAT)]
    pub(crate) position: [f32; 2],
    /// The id of the entity. Used and overriden when calling `vera::Vera::set()`
    #[format(R32_UINT)]
    pub(crate) entity_id: u32,
}
impl Veratex {
    /// A new `Veratex` with (x, y) 2D coordinates.
    pub fn new(x: f32, y: f32) -> Self {
        Veratex { position: [x, y], ..Default::default() }
    }
}
impl Default for Veratex {
    fn default() -> Veratex {
        Veratex {
            position: [-0.5, 0.5],
            entity_id: 0,
        }
    }
}

/// Any shape.  
/// 1 shape = 1 entity.  
/// This is what `fn new()` of specific shapes return.  
/// - `vertices` are the vertices of the shape, each group of three `Veratex` forming a triangle.
pub struct Shape {
    pub vertices: Vec<Veratex>,
}
impl Shape {
    /// Merges several shapes into one single entity, with empty transformations.
    pub fn from_merge(shapes: Vec<Shape>) -> Self {
        Self { 
            vertices: shapes
                .into_iter()
                .flat_map(move |shape| shape.vertices.into_iter())
                .collect(),
            ..Default::default()
        }
    }
    /// Creates a single entity from the given vertices, with empty transformations.
    pub fn from_vertices(vertices: Vec<Veratex>) -> Self {
        Self { 
            vertices,
            ..Default::default()
        }
    }
}
impl Default for Shape {
    fn default() -> Shape {
        Shape { 
            vertices: vec![
                Veratex::new(1.0, 0.5),
                Veratex::new(1.0, 1.0),
                Veratex::new(0.5, 1.0),
            ]
        }
    }
}

/// A triangle shape
pub struct Triangle;

impl Triangle {
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32, x3: f32, y3: f32) -> Shape {
        Shape::from_vertices(vec![
            Veratex::new(x1, y1),
            Veratex::new(x2, y2),
            Veratex::new(x3, y3),
        ])
    }
}

/// A rectangle shape
pub struct Rectangle;

impl Rectangle {
    pub fn new(width: f32, height: f32) -> Shape {
        Shape::from_merge(vec![
            Triangle::new(-width/2., -height/2., width/2., -height/2., -width/2., height/2.),
            Triangle::new(width/2., height/2., width/2., -height/2., -width/2., height/2.),
        ])
    }
}