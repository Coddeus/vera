pub trait Vertex {
    fn new(x: f32, y: f32) -> Self;
}

/// Any shape.  
/// 1 shape = 1 entity.  
/// This is what `fn new()` of specific shapes return.  
/// - `vertices` are the vertices of the shape, each group of three `Veratex` forming a triangle.
pub struct Shape<V: Vertex> {
    pub vertices: Vec<V>,
}
impl<V: Vertex> Shape<V> {
    /// Merges several shapes into one single entity, with empty transformations.
    pub fn from_merge(shapes: Vec<Shape<V>>) -> Self {
        Self { 
            vertices: shapes
                .into_iter()
                .flat_map(move |shape| shape.vertices.into_iter())
                .collect(),
            ..Default::default()
        }
    }
    /// Creates a single entity from the given vertices, with empty transformations.
    pub fn from_vertices(vertices: Vec<V>) -> Self {
        Self { 
            vertices,
            ..Default::default()
        }
    }
}
impl<V: Vertex> Default for Shape<V> {
    fn default() -> Shape<V> {
        Shape { 
            vertices: vec![
                V::new(1.0, 0.5),
                V::new(1.0, 1.0),
                V::new(0.5, 1.0),
            ]
        }
    }
}

/// A triangle shape
pub struct Triangle;

impl Triangle {
    pub fn new<V: Vertex>(x1: f32, y1: f32, x2: f32, y2: f32, x3: f32, y3: f32) -> Shape<V> {
        Shape::from_vertices(vec![
            V::new(x1, y1),
            V::new(x2, y2),
            V::new(x3, y3),
        ])
    }
}

/// A rectangle shape
pub struct Rectangle;

impl Rectangle {
    pub fn new<V: Vertex>(width: f32, height: f32) -> Shape<V> {
        Shape::from_merge(vec![
            Triangle::new(-width/2., -height/2., width/2., -height/2., -width/2., height/2.),
            Triangle::new(width/2., height/2., width/2., -height/2., -width/2., height/2.),
        ])
    }
}