/// Data for the transformation of a single vertex or shape.
/// You can have several transformations happening simultaneously, but the order of the transformations is likely important.
pub struct Tf {
    /// The type & value of the transformation.
    pub t: Transformation,
    /// The speed evolution of the transformation.
    pub e: Evolution,
    /// The start time of the transformation.
    pub start: f32,
    /// The duration of the transformation.
    pub end: f32,
}

/// The available transformations, used for vertices and shapes.
pub enum Transformation {
    /// A scale operation with the provided X, Y and Z scaling.
    Scale(f32, f32, f32),
    /// A translate operation with the provided X, Y and Z scaling.
    Translate(f32, f32, f32),
    /// A rotate operation around the X axis with the provided counter-clockwise angle, in radians.
    RotateX(f32),
    /// A rotate operation around the Y axis with the provided counter-clockwise angle, in radians.
    RotateY(f32),
    /// A rotate operation around the Z axis with the provided counter-clockwise angle, in radians.
    RotateZ(f32),
}

/// The speed evolution of per-vertex & per-shape transformations.
#[derive(Copy, Clone)]
pub enum Evolution {
    /// Constant speed from start to end.
    Linear,
}