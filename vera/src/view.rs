use crate::{
    Evolution, Transformation, Tf,
    D_TRANSFORMATION_SPEED_EVOLUTION, D_TRANSFORMATION_START_TIME, D_TRANSFORMATION_END_TIME
};

/// A view (a view represents the position, direction and angle of a camera).
/// - `t` are the runtime transformations of the view.
pub struct View {
    pub t: Vec<Tf>,
}

impl View {
    pub fn new() -> Self {
        Self {
            t: vec![]
        }
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