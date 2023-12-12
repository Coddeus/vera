//! Vertex creation and transformation.
//! This is an intermediate structure later reinterpreted by the Vera core.

use fastrand::f32;

use crate::{
    Evolution, ModelTransformation, ModelT,
    D_TRANSFORMATION_SPEED_EVOLUTION, D_TRANSFORMATION_START_TIME, D_TRANSFORMATION_END_TIME
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
    /// A unique identifier for this vertex.
    pub vertex_id: u32,
    /// An identifier for the shape this vertex belongs to.
    pub entity_id: u32,
    /// The position of the vertex.
    pub position: [f32; 3],
    /// The color of the vertex, in RGBA format.
    pub color: [f32; 4],

    // Treated in CPU
    pub t: Vec<ModelT>
}
impl Vertex {
    /// A new default Vertex. Call this method to initialize a vertex, before transforming it.

    /// # Don't
    /// DO NOT call this function in multithreaded scenarios, as it calls static mut. See [the crate root](super).
    pub fn new() -> Self {
        unsafe {
            Self {
                vertex_id: 0,
                entity_id: 0,
                position: super::D_VERTEX_POSITION,
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
            }
        }
    }

    /// Modifies the position of the vertex to (x, y, z).
    pub fn pos(mut self, x: f32, y: f32, z: f32) -> Self {
        self.position = [x, y, z];
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

    /// Sets the position of the vertex and ends the method calls pipe.
    pub(crate) fn set_pos(&mut self, x: f32, y: f32, z: f32) {
        self.position = [x, y, z];
    }

    /// Sets the color of the vertex and ends the method calls pipe.
    pub(crate) fn set_color(&mut self, red: f32, green: f32, blue: f32) {
        self.color = [red, green, blue, self.color[3]];
    }

    /// Sets the alpha of the vertex and ends the method calls pipe.
    pub(crate) fn set_alpha(&mut self, alpha: f32) {
        self.color[3] = alpha;
    }

    /// Adds a new transformation with default speed evolution, start time and end time.
    /// # Don't
    /// DO NOT call this function in multithreaded scenarios, as it calls static mut. See [the crate root](super).
    pub fn transform(mut self, transformation: ModelTransformation) -> Self {
        self.t.push(ModelT {
            t: transformation,
            e: unsafe { D_TRANSFORMATION_SPEED_EVOLUTION },
            start: unsafe { D_TRANSFORMATION_START_TIME },
            end: unsafe { D_TRANSFORMATION_END_TIME },
        });
        self
    }

    /// Modifies the speed evolution of the latest transformation added.
    pub fn evolution(mut self, e: Evolution) -> Self {
        self.t.last_mut().unwrap().e = e;
        self
    }

    /// Modifies the start time of the latest transformation added.
    pub fn start(mut self, start: f32) -> Self {
        self.t.last_mut().unwrap().start = start;
        self
    }

    /// Modifies the end time of the latest transformation added.
    pub fn end(mut self, end: f32) -> Self {
        self.t.last_mut().unwrap().end = end;
        self
    }
}
