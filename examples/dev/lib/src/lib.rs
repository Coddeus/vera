use std::{vec, f32::consts::PI};

use vera_shapes::{
    Input, View, Projection, Model, Triangle, Vertex, Transformation, 
    D_RANDOM_VERTEX_COLOR, D_VERTEX_ALPHA, D_VERTEX_COLOR,
};

#[no_mangle]
pub fn get() -> Input {
    unsafe {
        D_RANDOM_VERTEX_COLOR = true;
        D_VERTEX_COLOR = [1.0, 1.0, 1.0];
        D_VERTEX_ALPHA = 0.8;
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
                Vertex::new().pos(0.5, 0.5, 0.5)
                .transform(Transformation::Translate(-0.5, -0.5, 0.0)).start(0.0).end(1.0)
                .transform(Transformation::Translate(0.5, 0.5, 0.0)).start(1.0).end(2.0)
                .transform(Transformation::RotateZ(PI * 2.0)).start(0.0).end(2.0),
                Vertex::new().pos(0.0, 1.0, 0.0),
            ]),
            t1.transform(Transformation::Scale(2.0, 2.0, 2.0))
        ],
        v: View::new(),
        p: Projection::new(),
    }
}
