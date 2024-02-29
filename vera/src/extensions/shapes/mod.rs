use crate::{Model, ToModel, Vertex};

/// A triangle model.
pub struct Triangle {
    v1: Vertex,
    v2: Vertex,
    v3: Vertex,
}

impl Triangle {
    /// A new triangle with these 3 (x, y, z) vertices.
    pub fn new(x1: f32, y1: f32, z1: f32, x2: f32, y2: f32, z2: f32, x3: f32, y3: f32, z3: f32) -> Triangle {
        Self {
            v1: Vertex::new().pos(x1, y1, z1),
            v2: Vertex::new().pos(x2, y2, z2),
            v3: Vertex::new().pos(x3, y3, z3),
        }
    }
    /// A new triangle from the 3 given vertices.
    pub fn from_vertices(v1: Vertex, v2: Vertex, v3: Vertex) -> Triangle {
        Self {
            v1,
            v2,
            v3,
        }
    }
}

impl ToModel for Triangle {
    fn model(self) -> Model {
        Model::from_vertices(vec![self.v1, self.v2, self.v3])
    }
}