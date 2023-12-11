//! Crate root.
//!
//! Contains modifyable global variables (inside unsafe blocks) to choose Default behaviours. These variables start with `D_`.
//!
//! Everything further in this crate is reexported to this level.
//!
//! ## Don't
//! DO NOT modify/read these global variables simutaneously on different threads.  
//! Most likely, don't bother creating several threads at all. The vera core crate will do the performance job.
//! In case you really want multithreading: some function modify/read these variables, but aren't `unsafe` to simplify scripting. Check the docs of the functions you use to know which ones you should be careful with.

/// Default behaviour: which position to give vertices.
pub static mut D_VERTEX_POSITION: [f32; 3] = [0.0, 0.0, 0.0];
/// Default behaviour: which color to give vertices.
pub static mut D_VERTEX_COLOR: [f32; 3] = [0.0, 0.0, 0.0];
/// Default behaviour: What transparency value.
pub static mut D_VERTEX_ALPHA: f32 = 0.8;
/// Default behaviour: whether or not to choose random colors for each vertex.
/// Overrides `D_VERTEX_COLOR`, but not `D_VERTEX_ALPHA`.
pub static mut D_RANDOM_VERTEX_COLOR: bool = true;

mod vertex;
pub use vertex::*;
mod shape;
pub use shape::*;