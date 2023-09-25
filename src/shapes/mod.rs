pub mod regular_polygon;
pub mod textures;

/// Common functions across shapes.
/// Animation functions take "start" and "end" as time parameters. Set both to the same value to instantly transform the shape.
pub trait Transform {
    /// From `start` time to `end` time, following `ts` transition speed. Moves from the current position to `pos`, along the shape of `curve`.
    fn position(
        &mut self,
        start: f32,
        end: f32,
        ts: TransitionSpeed,
        pos: (f32, f32),
        curve: TransitionCurve,
    );
    /// From `start` time to `end` time, following `ts` transition speed. Rotates `angle` radians `clockwise`.
    fn rotation(
        &mut self,
        start: f32,
        end: f32,
        ts: TransitionSpeed,
        angle: f32,
        clockwise: bool,
    );
    /// From `start` time to `end` time, following `ts` transition speed. Scales the current shape by `scale`.
    fn size(&mut self, start: f32, end: f32, ts: TransitionSpeed, scale: f32);
    /// From `start` time to `end` time, following `ts` transition speed. Gradually changes the color to `color`.
    fn color(&mut self, start: f32, end: f32, ts: TransitionSpeed, color: Rgb);
}

/// Turns shape data into vertices and indices for Vulkan rendering
pub trait Vertind {
    fn vertind(&self) -> (Vec<f32>, Vec<u32>);
}

/// Transformation presets
pub enum Transformation {
    Move,
    Rotate,
    Scale,
    Color
}

/// Transition timing presets
pub enum TransitionSpeed {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
}

pub enum TransitionCurve {
    Linear,
    Sine,
    Circular((f32, f32), bool),
    QuadraticBezier((f32, f32)),
    CubicBezier((f32, f32), (f32, f32)),
}

pub struct Rgb(f32, f32, f32);
