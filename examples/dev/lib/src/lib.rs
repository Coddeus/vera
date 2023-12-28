use std::f32::consts::PI;

use fastrand;

use vera::{
    Input, View, Projection, Model, Vertex, Transformation, MetaInput, Evolution,
};

#[no_mangle]
fn get() -> Input {
    Input {
        meta: MetaInput {
            bg: [0.0, 0.0, 0.0, 1.0],
            start: 0.0,
            end: 5.0,
        },
        m: vec![
            Model::from_models(
                vec![
                    Model::from_models(vec![ // Combining transformed models needs vertex transformations - Matrix class: Vera-core => Vera, + user-visible?
                        triangle().transform(Transformation::Translate(-1.0, 0.0, 0.0)).start_t(0.0).end_t(0.0).transform(Transformation::RotateZ(PI * 0.5)).start_t(0.0).end_t(0.0),
                        triangle().transform(Transformation::RotateZ(PI)).start_t(0.0).end_t(0.0),
                        triangle().transform(Transformation::RotateZ(PI * 1.5)).start_t(0.0).end_t(0.0),
                    ])
                ]
            )
        ],
        v: View::new().transform(Transformation::Lookat(0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)).start_t(0.0).end_t(0.0),
        p: Projection::new().transform(Transformation::Perspective(-1.0, 1.0, -1.0, 1.0, 2.0, 10.0)).start_t(0.0).end_t(0.0),
    }
}

fn triangle() -> Model {
    Model::from_vertices(vec![
        Vertex::new().pos(0.0, 0.0, 0.0),
        Vertex::new().pos(1.0, 0.0, 0.0),
        Vertex::new().pos(0.5, 3.0f32.sqrt() / 2.0, 0.0),
    ]).rgb(0.0, 0.0, (fastrand::f32()+1.0)/2.0)
}

fn polygon(n: u16) -> Model {
    Model::from_vertices(vec![
        Vertex::new().pos(0.0, 0.0, 0.0),
        Vertex::new().pos(1.0, 0.0, 0.0),
        Vertex::new().pos(0.5, 3.0f32.sqrt() / 2.0, 0.0),
    ]).rgb(0.0, 0.0, (fastrand::f32()+1.0)/2.0)
}