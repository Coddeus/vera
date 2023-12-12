use crate::{
    Evolution, ViewTransformation, ViewT,
    D_TRANSFORMATION_SPEED_EVOLUTION, D_TRANSFORMATION_START_TIME, D_TRANSFORMATION_END_TIME
};

/// A view (a view represents the position, direction and angle of a camera).
/// - `t` are the runtime transformations of the view.
pub struct View {
    pub t: Vec<ViewT>,
}

impl View {
    /// Adds a new transformation with default speed evolution, start time and end time.
    /// # Don't
    /// DO NOT call this function in multithreaded scenarios, as it calls static mut. See [the crate root](super).
    pub fn transform(mut self, transformation: ViewTransformation) -> Self {
        self.t.push(ViewT {
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