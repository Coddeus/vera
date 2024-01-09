/// The evolution of a transformation or colorization.
#[derive(Copy, Clone)]
pub enum Evolution {
    /// Constant speed from start to end.
    Linear,
    /// Fast at the beginning, slow at the end.
    FastIn,
    /// Fast at the beginning, slow at the end.
    SlowOut,
    /// Slow at the beginning, fast at the end.
    FastOut,
    /// Slow at the beginning, fast at the end.
    SlowIn,
    /// Slow at ends, fast in middle.
    FastMiddle,
    /// Slow at ends, fast in middle.
    SlowInOut,
    /// Fast at ends, slow in middle.
    FastInOut,
    /// Fast at ends, slow in middle.
    SlowMiddle,
}

// -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// Data for the transformation of a single vertex or model.
/// 
/// ⚠ The transformations logic order is the order they are added to the vertex or model. Rotation R after translation T will not result in the same thing as T after R (if non-null).  
/// It may be different from the order in which they are applied, which depends on the start and end times of each transformation.  
/// You can have several transformations happening simultaneously.
pub struct Tf {
    /// The type & value of the transformation.
    pub t: Transformation,
    /// The speed evolution of the transformation.
    pub e: Evolution,
    /// The start time of the transformation.
    pub start: f32,
    /// The end time of the transformation.
    pub end: f32,
}

/// The available transformations.  
/// Their doc is prefixed with their general use case: Vertex/Model, View, Projection.  
/// Their doc lists the parameters as capital letters in the order they should be given.
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

    /// View: A full definition of the camera with the X, Y and Z eye position (at which point the camera is), the X, Y and Z target position (which point the camera stares at), and an X, Y and Z "up" vector (to determine the camera roll).
    Lookat(f32, f32, f32, f32, f32, f32, f32, f32, f32),
    // /// View: 
    // Move(f32, f32, f32),
    // /// View: 
    // Pitch(f32),
    // /// View: 
    // Yaw(f32),
    // /// View: 
    // Roll(f32),
    // 
    // /// Projection: 
    // Orthographic(f32, f32, f32, f32, f32, f32),
    /// Projection: A perspective projection with a near screen with the L left limit, R right limit, B bottom limit and T top limit, at a distance N from the camera and a far screen at a distance F from the camera.
    Perspective(f32, f32, f32, f32, f32, f32),
}

// -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// Data for the colorization of a single vertex.
/// 
/// ⚠ The colorization logic order is the order they are added to the vertex or model.  
/// It may be different from the order in which they are applied, which depends on the start and end times of each transformation.  
/// You can have several transformations happening simultaneously.
pub struct Cl {
    /// The type & value of the colorization.
    pub c: Colorization,
    /// The speed evolution of the colorization.
    pub e: Evolution,
    /// The start time of the colorization.
    pub start: f32,
    /// The end time of the colorization.
    pub end: f32,
}

/// The available colorizations.  
/// Their doc lists the parameters as capital letters in the order they should be given.
#[derive(Clone, Copy)]
pub enum Colorization {
    /// Changes the current color to this new RGBA color with rgba interpolation.
    ToColor(f32, f32, f32, f32),
}