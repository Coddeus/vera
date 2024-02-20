use crate::{Model, ToModel, Vertex};

pub struct Text {
    string: String,
    size: f32,
}

impl ToModel for Text {
    fn model(self) -> Model {
        Model::from_vertices(vec![
            Vertex::new(),
            Vertex::new(),
            Vertex::new(),
        ])
    }
}