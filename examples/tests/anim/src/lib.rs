use std::f32::consts::PI;

use vera::{Input, MetaInput, Model, Vertex, Transformation, View, Projection, Colorization};

#[no_mangle]
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
                    Vertex::new().pos(0.5, -0.5, 0.0).transform(Transformation::RotateZ(2.0*PI)).start_t(4.0).end_t(9.0),
                    Vertex::new().pos(-0.5, -0.5, 0.0).transform(Transformation::RotateZ(4.0*PI)).start_t(5.0).end_t(8.0),
                    Vertex::new().pos(0.0, -0.8, 0.0).transform(Transformation::RotateZ(8.0*PI)).start_t(6.0).end_t(7.0),
                ],
            ),//.transform(Transformation::RotateZ(4.0*PI)).start_t(0.0).end_t(1.0),
            Model::from_models(
                vec![
                    Model::from_vertices(vec![
                        Vertex::new().pos(0.0, 0.0, 0.0),
                        Vertex::new().pos(0.5, -0.5, 0.0),
                        Vertex::new().pos(0.5, 0.5, 0.0),
                    ]),
                    Model::from_vertices(vec![
                        Vertex::new().pos(0.0, 0.0, 0.0).recolor(Colorization::ToColor(0.0, 0.0, 0.0, 1.0)).start_c(1.0).end_c(10.0),
                        Vertex::new().pos(0.5, -0.5, 0.0),
                        Vertex::new().pos(-0.5, -0.5, 0.0),
                    ]),
                ],
            ),//.transform(Transformation::RotateZ(4.0*PI)).start_t(0.0).end_t(1.0),
            Model::from_vm(
                vec![
                    Vertex::new().pos(0.5, 0.5, 0.0),
                    Vertex::new().pos(-0.5, 0.5, 0.0),
                ],
                vec![
                    Model::from_vertices(vec![
                        Vertex::new().pos(-0.5, -0.5, 0.0),
                    ]).transform(Transformation::RotateZ(2.0*PI)).start_t(0.0).end_t(4.0),
                    Model::from_vertices(vec![
                    ]),
                ],
            ).transform(Transformation::RotateZ(2.0*PI)).start_t(0.0).end_t(4.0)
        ]).transform(Transformation::RotateZ(2.0*PI)).start_t(0.0).end_t(4.0)],
        v: View::new(),
        p: Projection::new(),
    }
}