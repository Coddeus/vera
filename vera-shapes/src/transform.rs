/// The speed evolution of a transformation.
#[derive(Copy, Clone)]
pub enum Evolution {
    /// Constant speed from start to end.
    Linear,

    Unimplemented,
}

// -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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

/// The available transformations
/// Their doc is prefixed with their general use case: Vertex/Model, View, Projection.
#[derive(Clone, Copy)]
pub enum Transformation {
    /// Vertex/Model: A scale operation with the provided X, Y and Z scaling.
    Scale(f32, f32, f32),
    /// Vertex/Model: A translate operation with the provided X, Y and Z scaling.
    Translate(f32, f32, f32),
    /// Vertex/Model: A rotate operation around the X axis with the provided counter-clockwise angle, in radians.
    RotateX(f32),
    /// Vertex/Model: A rotate operation around the Y axis with the provided counter-clockwise angle, in radians.
    RotateY(f32),
    /// Vertex/Model: A rotate operation around the Z axis with the provided counter-clockwise angle, in radians.
    RotateZ(f32),

    /// View: 
    Lookat(f32, f32, f32, f32, f32, f32, f32, f32, f32),
    /// View: 
    Move(f32, f32, f32),
    /// View: 
    Pitch(f32),
    /// View: 
    Yaw(f32),
    /// View: 
    Roll(f32),

    /// Projection: 
    Orthographic(f32, f32, f32, f32, f32, f32),
    /// Projection: 
    Perspective(f32, f32, f32, f32, f32, f32),
}