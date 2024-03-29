//! Crate root.
//!
//! Contains modifyable global variables (inside unsafe blocks) to choose Default behaviours. These variables start with `D_`.
//!
//! Everything further in this crate is reexported to this level.
//!
//! ## Don't
//! DO NOT modify/read these global variables simutaneously on different threads.  
//! Most likely, don't bother creating several threads at all. The Vera core crate will do the performance job.
//! In case you really want multithreading: some function modify/read these variables, but aren't `unsafe` to simplify scripting. Check the docs of the functions you use to know which ones you should be careful with.

/// Default behaviour: whether or not to choose random colors for each vertex.
/// Overrides `D_VERTEX_COLOR`, but not `D_VERTEX_ALPHA`.
pub static mut D_RANDOM_VERTEX_COLOR: bool = true;
/// Default behaviour: Transformation speed evolution.
pub static mut D_TRANSFORMATION_SPEED_EVOLUTION: Evolution = Evolution::Linear;
/// Default behaviour: Transformation start time.
pub static mut D_TRANSFORMATION_START_TIME: f32 = 0.0;
/// Default behaviour: Transformation end time.
pub static mut D_TRANSFORMATION_END_TIME: f32 = 2.0;
/// Default behaviour: Colorization speed evolution.
pub static mut D_COLORIZATION_SPEED_EVOLUTION: Evolution = Evolution::Linear;
/// Default behaviour: Colorization start time.
pub static mut D_COLORIZATION_START_TIME: f32 = 0.0;
/// Default behaviour: Colorization end time.
pub static mut D_COLORIZATION_END_TIME: f32 = 2.0;
/// Default behaviour: which position to give vertices.
pub static mut D_VERTEX_POSITION: [f32; 3] = [0.0, 0.0, 0.0];
/// Default behaviour: which color to give vertices.
pub static mut D_VERTEX_COLOR: [f32; 3] = [0.0, 0.0, 0.0];
/// Default behaviour: What transparency value to give vertices.
pub static mut D_VERTEX_ALPHA: f32 = 1.0;

/// Default view matrix
pub static mut VIEW: f32 = 0.8;

/// A vertex, belonging to a model
mod vertex;
pub use vertex::*;
/// A model, something that is drawn
mod model;
pub use model::*;
/// A view, representation of a camera
mod view;
pub use view::*;
/// A projection, defines the viewing frustrum.
mod projection;
pub use projection::*;
/// Transformations for vertices/models, views and projections.
mod transform;
pub use transform::*;

/// Extensions, to avoid boilerplate in some cases.
mod extensions;
pub use extensions::*;

/// The input of the Vera core. This is what you send when calling functions like `create()` or `reset()`.
/// It contains everything that will be drawn and updated.
pub struct Input {
    /// Metadata for the animation
    pub meta: MetaInput,
    /// All models, with their transformations.
    pub m: Vec<Model>,
    /// The view, with its transformations
    pub v: View,
    /// The projection, with its transformations
    // /// By default, the visible 3D frustrum corresponds to Transformation::Orthographic(-1.0, 1.0, -1.0, 1.0, 1.0, -1.0)
    pub p: Projection,
}

/// Meta-information about the animation
pub struct MetaInput {
    /// The background color.
    pub bg: [f32 ; 4],
    /// The instant the animation starts.
    pub start: f32,
    /// The instant the animation end.
    pub end: f32,
}
impl Default for MetaInput {
    fn default() -> Self {
        MetaInput {
            bg: [0.3, 0.3, 0.3, 1.0],
            start: 0.0,
            end: 5.0,
        }
    }
}