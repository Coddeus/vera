use std::time::Instant;

/// Data for a single vertex or shape transformation.
/// Transformations are applied one at a time, by 
/// You can have several transformations happening simultaneously, but the order of the transformations is likely important.
pub struct Tf {
    /// The type & value of the transformation
    t: Transformation,
    /// The speed evolution of the transformation
    e: Evolution,
    /// The start time of the transformation
    start: Instant,
    /// The duration of the transformation
    duration: f32,
}

/// The available transformations, used for vertices and shapes.
pub enum Transformation {
    Scale(f32, f32, f32),
    Translate(f32, f32, f32),
    RotateX(f32),
    RotateY(f32),
    RotateZ(f32),
}

/// The speed evolution of per-vertex & per-shape transformations
pub enum Evolution {
    /// Constant speed from start to end
    Linear,
}