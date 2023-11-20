/// A vertex with:
/// - a 2D position
/// - the id of the entity it belongs to
/// - (A Vec of transformations)
pub struct Vertex {
    pub position: [f32 ; 2],
    pub entity_id: u32,
}
impl Vertex {
    /// A new Vertex with (x, y) 2D coordinates.
    fn new(x: f32, y: f32) -> Self {
        Vertex { position: [x, y], ..Default::default() }
    }
}
impl Default for Vertex {
    fn default() -> Vertex {
        Vertex {
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
    pub vertices: Vec<Vertex>,
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
    pub fn from_vertices(vertices: Vec<Vertex>) -> Self {
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
                Vertex::new(1.0, 0.5),
                Vertex::new(1.0, 1.0),
                Vertex::new(0.5, 1.0),
            ]
        }
    }
}

/// A triangle shape
pub struct Triangle;

impl Triangle {
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32, x3: f32, y3: f32) -> Shape {
        Shape::from_vertices(vec![
            Vertex::new(x1, y1),
            Vertex::new(x2, y2),
            Vertex::new(x3, y3),
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