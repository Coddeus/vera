use super::*;

/// A regular polygon shape. Can be used to render "circles" by providing many sides.
pub struct RegularPolygon {
    // /// The number of sides of the polygon.
    // sides: u16,
    // /// The radius of the circle around the polygon. Defaults to `100.`.
    // radius: f32,
    // /// The rotation of the polygon in radians, counter-clockwise. Defaults to 0, with a point aligned horizontally with the center, on its right.
    // rotation: f32,
    // /// The position of the center of the polygon. Defaults to the center of the (100, 100).
    // position: (f32, f32),
    // /// The color of the polygon
    // color: Rgb,
    vert: Vec<f32>,
    ind: Vec<u32>,
}

impl RegularPolygon {
    pub fn new(sides: u16) -> Self {
        let vert = 
        RegularPolygon {
            sides,
            radius: 100.,
            rotation: 100.,
            position: (100., 100.),
            color: Rgb(0.9, 0.2, 0.2),
        }
    }
}

impl Vertind for RegularPolygon {
    fn vertind(&self) -> (Vec<f32>, Vec<u32>) {
        let vert: (f32, f32, f32, f32, f32) = (self.position.0, self.position.1, self.color.0, self.color.1, self.color.2);
        let ind: 
        for 
    }
}