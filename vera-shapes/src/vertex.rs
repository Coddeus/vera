use fastrand::f32;

/// A vertex with:
/// - the id of the entity it belongs to. Changing it manually has no effect;
/// - a 3D XYZ position;
/// - an RGBA color;
/// - (A Vec of transformations);
///
/// Vertices enable building Triangles, which enable building all other shapes.
pub struct Vertex {
    pub entity_id: u32,
    pub position: [f32; 3],
    pub color: [f32; 4],
}
impl Vertex {
    /// A new default Vertex. Call this method to initialize a vertex, before transforming it.
    ///
    /// Initializes according to the root variables [D_VERTEX_POSITION](super::D_VERTEX_POSITION), [D_VERTEX_COLOR](super::D_VERTEX_COLOR), [D_VERTEX_ALPHA](super::D_VERTEX_ALPHA), [D_RANDOM_COLORS](super::D_RANDOM_COLORS).

    /// # Don't
    /// DO NOT call this function in multithreaded scenarios. See [the crate root](super)
    pub fn new() -> Self {
        unsafe {
            Self {
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
            }
        }
    }

    pub fn pos(mut self, x: f32, y: f32, z: f32) -> Self {
        self.position = [x, y, z];
        self
    }

    pub fn r(mut self, red: f32) -> Self {
        self.color[0] = red;
        self
    }

    pub fn g(mut self, green: f32) -> Self {
        self.color[1] = green;
        self
    }

    pub fn b(mut self, blue: f32) -> Self {
        self.color[2] = blue;
        self
    }

    pub fn a(mut self, alpha: f32) -> Self {
        self.color[3] = alpha;
        self
    }

    pub fn color(mut self, red: f32, green: f32, blue: f32) -> Self {
        self.color = [red, green, blue, self.color[3]];
        self
    }

    pub fn alpha(mut self, alpha: f32) -> Self {
        self.color[3] = alpha;
        self
    }

    pub fn set_pos(&mut self, x: f32, y: f32, z: f32) {
        self.position = [x, y, z];
    }

    pub fn set_color(&mut self, red: f32, green: f32, blue: f32) {
        self.color = [red, green, blue, self.color[3]];
    }

    pub fn set_alpha(&mut self, alpha: f32) {
        self.color[3] = alpha;
    }
}
