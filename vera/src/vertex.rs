//! Vertex creation and transformation.
//! This is an intermediate structure later reinterpreted by the Vera core.

use fastrand::f32;

use crate::{
    Evolution, Tf, Cl, Transformation, Colorization, 
    D_TRANSFORMATION_SPEED_EVOLUTION, D_TRANSFORMATION_START_TIME, D_TRANSFORMATION_END_TIME,
    D_COLORIZATION_SPEED_EVOLUTION, D_COLORIZATION_START_TIME, D_COLORIZATION_END_TIME,
};

/// A vertex with:
/// - its id. Changing it manually has no effect;
/// - the id of the entity it belongs to. Changing it manually has no effect;
/// - a 3D XYZ position;
/// - an RGBA color;
/// - (A Vec of transformations);
///
/// Vertices enable building Triangles, which enable building all other shapes.
pub struct Vertex {
    // Sent to GPU
    /// The position of the vertex.
    pub position: [f32; 4],
    /// The color of the vertex, in RGBA format.
    pub color: [f32; 4],

    // Treated in CPU
    pub t: Vec<Tf>,
    pub c: Vec<Cl>,
}
impl Vertex {
    /// A new default Vertex. Call this method to initialize a vertex, before transforming it.

    /// # Don't
    /// DO NOT call this function in multithreaded scenarios, as it calls static mut. See [the crate root](super).
    pub fn new() -> Self {
        unsafe {
            Self {
                position: [
                    super::D_VERTEX_POSITION[0],
                    super::D_VERTEX_POSITION[1],
                    super::D_VERTEX_POSITION[2],
                    1.0,
                ],
                color: if super::D_RANDOM_VERTEX_COLOR {
                    [f32(), f32(), f32(), super::D_VERTEX_ALPHA]
                } else {
                    [
                        super::D_VERTEX_COLOR[0],
                        super::D_VERTEX_COLOR[1],
                        super::D_VERTEX_COLOR[2],
                        super::D_VERTEX_ALPHA,
                    ]
                },
                t: vec![],
                c: vec![],
            }
        }
    }

    /// Modifies the position of the vertex to (x, y, z).
    pub fn pos(mut self, x: f32, y: f32, z: f32) -> Self {
        self.position = [x, y, z, 1.0];
        self
    }

    /// Modifies the red color channel of the vertex color.
    pub fn r(mut self, red: f32) -> Self {
        self.color[0] = red;
        self
    }

    /// Modifies the green color channel of the vertex color.
    pub fn g(mut self, green: f32) -> Self {
        self.color[1] = green;
        self
    }

    /// Modifies the blue color channel of the vertex color.
    pub fn b(mut self, blue: f32) -> Self {
        self.color[2] = blue;
        self
    }

    /// Modifies the alpha color channel of the vertex color.
    pub fn a(mut self, alpha: f32) -> Self {
        self.color[3] = alpha;
        self
    }

    /// Modifies the red, green and blue color channels of the vertex color.
    pub fn rgb(mut self, red: f32, green: f32, blue: f32) -> Self {
        self.color = [red, green, blue, self.color[3]];
        self
    }

    /// Modifies the red, green, blue and alpha color channels of the vertex color.
    pub fn rgba(mut self, red: f32, green: f32, blue: f32, alpha: f32) -> Self {
        self.color = [red, green, blue, alpha];
        self
    }

    // /// Sets the position of the vertex and ends the method calls pipe.
    // pub(crate) fn set_pos(&mut self, x: f32, y: f32, z: f32) {
    //     self.position = [x, y, z];
    // }

    /// Sets the color of the vertex and ends the method calls pipe.
    pub(crate) fn set_rgb(&mut self, red: f32, green: f32, blue: f32) {
        self.color = [red, green, blue, self.color[3]];
    }

    /// Sets the alpha of the vertex and ends the method calls pipe.
    pub(crate) fn set_alpha(&mut self, alpha: f32) {
        self.color[3] = alpha;
    }

    // -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    /// Adds a new transformation to this vertex with default speed evolution, start time and end time.
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

    /// Adds a new color change to this vertex with default speed evolution, start time and end time.
    /// # Don't
    /// DO NOT call this function in multithreaded scenarios, as it calls static mut. See [the crate root](super).
    pub fn recolor(mut self, colorization: Colorization) -> Self {
        self.c.push(Cl {
            c: colorization,
            e: unsafe { D_COLORIZATION_SPEED_EVOLUTION },
            start: unsafe { D_COLORIZATION_START_TIME },
            end: unsafe { D_COLORIZATION_END_TIME },
        });
        self
    }

    /// Modifies the speed evolution of the latest colorization added.
    pub fn evolution_c(mut self, e: Evolution) -> Self {
        self.c.last_mut().unwrap().e = e;
        self
    }

    /// Modifies the start time of the latest colorization added.
    /// A start after an end will result in the colorization being instantaneous at start.
    pub fn start_c(mut self, start: f32) -> Self {
        self.c.last_mut().unwrap().start = start;
        self
    }

    /// Modifies the end time of the latest colorization added.
    /// An end before a start will result in the colorization being instantaneous at start.
    pub fn end_c(mut self, end: f32) -> Self {
        self.c.last_mut().unwrap().end = end;
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
