use crate::{
    Evolution, Tf, Transformation, Vertex, D_TRANSFORMATION_END_TIME,
    D_TRANSFORMATION_SPEED_EVOLUTION, D_TRANSFORMATION_START_TIME,
};

/// A model (a model is a shape).
/// 1 model = 1 entity.  
/// This is what `fn new()` of specific models return.  
/// - `vertices` are the vertices of the model, each group of three `Vertex` forming a triangle.
/// - `t` are the runtime transformations of the model.
pub struct Model {
    pub vertices: Vec<Vertex>,
    pub t: Vec<Tf>,
}

impl Model {
    /// Merges several models into one single entity, with empty transformations.
    pub fn from_merge(models: Vec<Model>) -> Self {
        Self {
            vertices: models
                .into_iter()
                .flat_map(move |model| model.vertices.into_iter())
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
    /// Sets the model color
    pub fn color(mut self, red: f32, green: f32, blue: f32) -> Self {
        self.vertices
            .iter_mut()
            .for_each(|v| v.set_color(red, green, blue));
        self
    }
    /// Sets the model opacity
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.vertices.iter_mut().for_each(|v| v.set_alpha(alpha));
        self
    }

    /// Adds a new transformation with default speed evolution, start time and end time.
    /// # Don't
    /// DO NOT call this function in multithreaded scenarios, as it calls static mut. See [the crate root](super).
    pub fn transform(mut self, transformation: Transformation) -> Self {
        self.t.push(Tf {
            t: transformation,
            e: unsafe { D_TRANSFORMATION_SPEED_EVOLUTION },
            start: unsafe { D_TRANSFORMATION_START_TIME },
            end: unsafe { D_TRANSFORMATION_END_TIME },
        });
        self
    }

    /// Modifies the speed evolution of the latest transformation added.
    pub fn evolution(mut self, e: Evolution) -> Self {
        self.t.last_mut().unwrap().e = e;
        self
    }

    /// Modifies the start time of the latest transformation added.
    pub fn start(mut self, start: f32) -> Self {
        self.t.last_mut().unwrap().start = start;
        self
    }

    /// Modifies the end time of the latest transformation added.
    pub fn end(mut self, end: f32) -> Self {
        self.t.last_mut().unwrap().end = end;
        self
    }
}

/// A triangle model
pub struct Triangle;

impl Triangle {
    pub fn new(v1: Vertex, v2: Vertex, v3: Vertex) -> Model {
        Model::from_vertices(vec![v1, v2, v3])
    }
}

// // // Better have all models (except the triangle) be fully user-defined.
// /// A rectangle model
// pub struct Rectangle;
//
// impl Rectangle {
//     pub fn new(width: f32, height: f32) -> model {
//         model::from_merge(vec![
//             Triangle::new(-width/2., -height/2., width/2., -height/2., -width/2., height/2.),
//             Triangle::new(width/2., height/2., width/2., -height/2., -width/2., height/2.),
//         ])
//     }
// }
