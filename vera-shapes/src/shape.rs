use crate::{Vertex, Transformation};

/// Any shape.  
/// 1 shape = 1 entity.  
/// This is what `fn new()` of specific shapes return.  
/// - `vertices` are the vertices of the shape, each group of three `Vertex` forming a triangle.
/// - `t` are the runtime transformations of the shape.
pub struct Shape {
    pub vertices: Vec<Vertex>,
    pub t: Vec<Transformation>
}
impl Shape {
    /// Merges several shapes into one single entity, with empty transformations.
    pub fn from_merge(shapes: Vec<Shape>) -> Self {
        Self { 
            vertices: shapes
                .into_iter()
                .flat_map(move |shape| shape.vertices.into_iter())
                .collect(),
            t: vec![],
        }
    }
    /// Creates a single entity from the given vertices, with empty transformations.
    /// The number of vertices should (most likely) be a multiple of 3.
    pub fn from_vertices(vertices: Vec<Vertex>) -> Self {
        Self { 
            vertices,
            t: vec![],
        }
    }
    /// Sets the shape color
    pub fn color(&mut self, red: f32, green: f32, blue: f32) {
        self.vertices.iter_mut().for_each(|v| v.set_color(red, green, blue))
    }
    /// Sets the shape opacity
    pub fn alpha(&mut self, alpha: f32) {
        self.vertices.iter_mut().for_each(|v| v.set_alpha(alpha))
    }
}

/// A triangle shape
pub struct Triangle;

impl Triangle {
    pub fn new(v1: Vertex, v2: Vertex, v3: Vertex) -> Shape {
        Shape::from_vertices(vec![v1, v2, v3])
    }
}

// // // Better have all shapes (except the triangle) be fully user-defined.
// /// A rectangle shape
// pub struct Rectangle;
// 
// impl Rectangle {
//     pub fn new(width: f32, height: f32) -> Shape {
//         Shape::from_merge(vec![
//             Triangle::new(-width/2., -height/2., width/2., -height/2., -width/2., height/2.),
//             Triangle::new(width/2., height/2., width/2., -height/2., -width/2., height/2.),
//         ])
//     }
// }