use crate::{Tf, D_TRANSFORMATION_SPEED_EVOLUTION, D_TRANSFORMATION_START_TIME, D_TRANSFORMATION_END_TIME, Evolution, Transformation};



/// A view (a view represents the view of ).
/// 1 model = 1 entity.  
/// This is what `fn new()` of specific models return.  
/// - `vertices` are the vertices of the model, each group of three `Vertex` forming a triangle.
/// - `t` are the runtime transformations of the model.
pub struct View {
    pub t: Vec<Tf>,
}

impl View {
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