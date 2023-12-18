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
        D_VERTEX_ALPHA = 1.0;
    }

    let t1 = Model::from_vertices(
        vec![
            Vertex::new().pos(0.5, 0.5, 0.0).rgb(1.0, 1.0, 1.0),
            Vertex::new().pos(-0.5, 0.5, 0.0).rgb(0.5, 0.5, 0.5),
            Vertex::new().pos(0.5, -0.5, 0.0).rgb(0.5, 0.5, 0.5),
            Vertex::new().pos(0.5, -0.5, 0.0).rgb(0.5, 0.5, 0.5),
            Vertex::new().pos(-0.5, 0.5, 0.0).rgb(0.5, 0.5, 0.5),
            Vertex::new().pos(-0.5, -0.5, 0.0).rgb(0.0, 0.0, 0.0),
        ]
    );

    Input {
        m: vec![
            t1.transform(Transformation::Scale(2.0, 2.0, 2.0)).start(0.0).end(2.0)
        ],
        v: View::new().transform(Transformation::Lookat(2.0, -2.0, -5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)).start(0.0).end(0.0),
        p: Projection::new().transform(Transformation::Perspective(-1.0, 1.0, -1.0, 1.0, 1.0, 10.0)).start(0.0).end(0.0),
    }
}
