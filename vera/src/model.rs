use crate::{
    Evolution, Tf, Cl, Transformation, Colorization, Vertex,
    D_TRANSFORMATION_SPEED_EVOLUTION, D_TRANSFORMATION_START_TIME, D_TRANSFORMATION_END_TIME,
    D_COLORIZATION_SPEED_EVOLUTION, D_COLORIZATION_START_TIME, D_COLORIZATION_END_TIME,
};

/// A model (a model is a shape).
/// 1 model = 1 entity.  
/// This is what `fn new()` of specific models return.  
/// - `models` are the models contained inside of this one, each group of three `Vertex` forming a triangle.
/// - `vertices` are the vertices of the model, each group of three `Vertex` forming a triangle.
/// - `t` are the runtime transformations of the model.
pub struct Model {
    pub models: Vec<Model>,
    pub vertices: Vec<Vertex>,
    pub t: Vec<Tf>,
}

impl Model {
    /// Merges several models into one single entity, with empty transformations.
    pub fn from_vm(vertices: Vec<Vertex>, models: Vec<Model>) -> Self {
        Self {
            models: vec![],
            vertices: models
                .into_iter()
                .flat_map(move |model| model.vertices.into_iter())
                .collect(),
            t: vec![],
        }
    }
    /// Merges several models into one single entity, with empty transformations.
    pub fn from_models(models: Vec<Model>) -> Self {
        Self {
            models,
            vertices: vec![],
            t: vec![],
        }
    }
    /// Creates a single entity from the given vertices, with empty transformations.
    /// The number of vertices should (most likely) be a multiple of 3.
    pub fn from_vertices(vertices: Vec<Vertex>) -> Self {
        Self {
            models: vec![],
            vertices,
            t: vec![],
        }
    }
    /// Sets the model rgb color values.
    pub fn rgb(mut self, red: f32, green: f32, blue: f32) -> Self {
        self.vertices
            .iter_mut()
            .for_each(|v| v.set_rgb(red, green, blue));
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

    /// Adds a new color change with default speed evolution, start time and end time.
    /// # Don't
    /// DO NOT call this function in multithreaded scenarios, as it calls static mut. See [the crate root](super).
    pub fn recolor(mut self, colorization: Colorization) -> Self {
        self.vertices.iter_mut().for_each(|v| { v.c.push(Cl {
            c: colorization,
            e: unsafe { D_COLORIZATION_SPEED_EVOLUTION },
            start: unsafe { D_COLORIZATION_START_TIME },
            end: unsafe { D_COLORIZATION_END_TIME },
        })});
        self
    }

    /// Modifies the speed evolution of the latest colorization added.
    pub fn evolution_c(mut self, e: Evolution) -> Self {
        self.vertices.iter_mut().for_each(|v| { v.c.last_mut().unwrap().e = e; });
        self
    }

    /// Modifies the start time of the latest colorization added.
    /// A start after an end will result in the colorization being instantaneous at start.
    pub fn start_c(mut self, start: f32) -> Self {
        self.vertices.iter_mut().for_each(|v| { v.c.last_mut().unwrap().start = start; });
        self
    }

    /// Modifies the end time of the latest colorization added.
    /// An end before a start will result in the colorization being instantaneous at start.
    pub fn end_c(mut self, end: f32) -> Self {
        self.vertices.iter_mut().for_each(|v| { v.c.last_mut().unwrap().end = end; });
        self
    }

    /// Modifies the speed evolution of the latest transformation added.
    pub fn evolution_t(mut self, e: Evolution) -> Self {
        self.t.last_mut().unwrap().e = e;
        self
    }

    /// Modifies the start time of the latest transformation added.
    /// A start after an end will result in the transformation being instantaneous at start.
    pub fn start_t(mut self, start: f32) -> Self {
        self.t.last_mut().unwrap().start = start;
        self
    }

    /// Modifies the end time of the latest transformation added.
    /// An end before a start will result in the transformation being instantaneous at start.
    pub fn end_t(mut self, end: f32) -> Self {
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