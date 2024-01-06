use std::f32::consts::PI;

use vera::{Input, MetaInput, Model, Vertex, Transformation, View, Projection};

pub(crate) fn get() -> Input {
    Input {
        meta: MetaInput {
            bg: [0.0, 0.0, 0.0, 1.0],
            start: 0.0,
            end: 13.0,
        },
        m: vec![ Model::from_models(vec![
            Model::from_vertices(
                vec![
                    Vertex::new().pos(0.5, -0.5, 0.0).transform(Transformation::RotateZ(2.0*PI)).start_t(2.0).end_t(7.0),
                    Vertex::new().pos(-0.5, -0.5, 0.0).transform(Transformation::RotateZ(4.0*PI)).start_t(3.0).end_t(6.0),
                    Vertex::new().pos(0.0, -0.8, 0.0).transform(Transformation::RotateZ(8.0*PI)).start_t(4.0).end_t(5.0),
                ],
            ),
            Model::from_models(
                vec![
                    Model::from_vertices(vec![
                        Vertex::new().pos(0.0, 0.0, 0.0),
                        Vertex::new().pos(0.5, -0.5, 0.0),
                        Vertex::new().pos(0.5, 0.5, 0.0),
                    ]),
                    Model::from_vertices(vec![
                        Vertex::new().pos(0.0, 0.0, 0.0),
                        Vertex::new().pos(0.5, -0.5, 0.0),
                        Vertex::new().pos(-0.5, -0.5, 0.0),
                    ]),
                ],
            ),
            Model::from_vm(
                vec![
                    Vertex::new().pos(0.5, 0.5, 0.0),
                    Vertex::new().pos(-0.5, 0.5, 0.0),
                ],
                vec![
                    Model::from_vertices(vec![
                        Vertex::new().pos(-0.5, -0.5, 0.0),
                    ]).transform(Transformation::RotateZ(4.0*PI)).start_t(8.0).end_t(9.0),
                    Model::from_vertices(vec![
                    ]),
                ],
            )
        ]).transform(Transformation::RotateZ(4.0*PI)).start_t(10.0).end_t(12.0)],
        v: View::new(),
        p: Projection::new(),
    }
}

