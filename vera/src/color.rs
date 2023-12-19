#[derive(Clone, Copy)]
pub(crate) struct Color(pub [f32 ; 4]);

impl Color {
    pub const WHITISH: Self = Self([0.9, 0.9, 0.9, 1.0]);   // Can be seen on (common) pure white backgrounds
    pub const BLACKISH: Self = Self([0.1, 0.1, 0.1, 1.0]);  // Can be seen on (common) pure black backgrounds

    /// Interpolates `self` with `color` with the (0.0 to 1.0) `advancement` value.
    pub(crate) fn interpolate(&mut self, color: [f32 ; 4], advancement: f32) {
        self.0 = [
            self.0[0] * (1.0-advancement) + color[0] * advancement,
            self.0[1] * (1.0-advancement) + color[1] * advancement,
            self.0[2] * (1.0-advancement) + color[2] * advancement,
            self.0[3] * (1.0-advancement) + color[3] * advancement,
        ];
    }
}