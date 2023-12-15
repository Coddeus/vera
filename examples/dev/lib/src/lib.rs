use std::vec;

use vera_shapes::{
    Input, View, Projection, Model, Triangle, Vertex,
    D_RANDOM_VERTEX_COLOR, D_VERTEX_ALPHA, D_VERTEX_COLOR,
};

#[no_mangle]
pub fn get() -> Input {
    unsafe {
        D_RANDOM_VERTEX_COLOR = false;
        D_VERTEX_COLOR = [1.0, 1.0, 1.0];
        D_VERTEX_ALPHA = 1.0;
    }

    let t1 = Triangle::new(
        Vertex::new().pos(0.0, 0.5, 0.0),
        Vertex::new().pos(0.3, 0.2, 0.0),
        Vertex::new().pos(-0.2, 0.4, 0.0),
    );

    Input {
        m: vec![
            Model::from_vertices(vec![
                Vertex::new().pos(0.0, 0.0, 0.0),
                Vertex::new().pos(0.5, 0.5, 0.5),
                Vertex::new().pos(0.0, 1.0, 0.0),
            ]),
            t1,
        ],
        v: View::new(),
        p: Projection::new(),
    }
}
