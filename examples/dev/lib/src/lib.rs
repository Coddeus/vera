use std::{vec, f32::consts::PI};
use fastrand;

use vera_shapes::{
    Input, View, Projection, Model, Triangle, Vertex, Transformation, 
    D_RANDOM_VERTEX_COLOR, D_VERTEX_ALPHA, D_VERTEX_COLOR, Colorization, MetaInput, Evolution,
};

#[no_mangle]
fn get() -> Input {
    Input {
        meta: MetaInput {
            bg: [0.0, 0.0, 0.0, 1.0],
            start: 0.0,
            end: 10.0,
        },
        m: (0..64)
            .map(|n| 
                Model::from_merge(
                    vec![
                        Model::from_vertices(vec![
                            Vertex::new().pos(1.0, 1.0, 1.0),
                            Vertex::new().pos(-1.0, -1.0, 1.0),
                            Vertex::new().pos(1.0, -1.0, -1.0),
                        ]).rgb(0.0, 0.0, (fastrand::f32()+1.0) / 2.0),
                        Model::from_vertices(vec![
                            Vertex::new().pos(1.0, 1.0, 1.0),
                            Vertex::new().pos(-1.0, 1.0, -1.0),
                            Vertex::new().pos(1.0, -1.0, -1.0),
                        ]).rgb(0.0, 0.0, (fastrand::f32()+1.0) / 2.0),
                        Model::from_vertices(vec![
                            Vertex::new().pos(1.0, 1.0, 1.0),
                            Vertex::new().pos(-1.0, 1.0, -1.0),
                            Vertex::new().pos(-1.0, -1.0, 1.0),
                        ]).rgb(0.0, 0.0, (fastrand::f32()+1.0) / 2.0),
                        Model::from_vertices(vec![
                            Vertex::new().pos(-1.0, -1.0, 1.0),
                            Vertex::new().pos(-1.0, 1.0, -1.0),
                            Vertex::new().pos(1.0, -1.0, -1.0),
                        ]).rgb(0.0, 0.0, (fastrand::f32()+1.0) / 2.0),
                    ]
                )
                .transform(Transformation::Scale(0.1, 0.1, 0.1)).start_t(0.0+n as f32/50.0).end_t(0.5+n as f32/50.0).evolution_t(Evolution::FastInOut)
                .transform(Transformation::Translate(((n/16) as f32 - 1.5) / 2.0, (((n%16)/4) as f32 - 1.5) / 2.0, ((n%4) as f32 - 1.5) / 2.0)).start_t(1.0+n as f32/50.0).end_t(1.5+n as f32/50.0).evolution_t(Evolution::FastIn)
            ).collect(),
        v: View::new()
            .transform(Transformation::Lookat(10.0, 5.0, -31.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)).start_t(0.0).end_t(0.0).evolution_t(Evolution::FastMiddle)
            // .transform(Transformation::Lookat(0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)).start_t(4.0).end_t(9.0).evolution_t(Evolution::Linear)
            // .transform(Transformation::Lookat(0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)).start_t(0.0).end_t(5.0)
            ,
        p: Projection::new()
            .transform(Transformation::Perspective(-1.0, 1.0, -1.0, 1.0, 29.0, 33.0)).start_t(0.0).end_t(0.0).evolution_t(Evolution::FastMiddle)
            .transform(Transformation::Perspective(-1.0, 1.0, -1.0, 1.0, 1.0, 32.0)).start_t(4.0).end_t(9.0).evolution_t(Evolution::Linear)
            // .transform(Transformation::Orthographic(-1.0, 1.0, -1.0, 1.0, 2.0, 30.0)).start_t(6.0).end_t(9.0)
            // .transform(Transformation::Orthographic(-1.0, 1.0, -1.0, 1.0, 0.0, -0.1)).start_t(4.5).end_t(5.0)
            // .transform(Transformation::Orthographic(-1.0, 1.0, -1.0, 1.0, -1.9, -2.0)).start_t(5.5).end_t(7.5)
            // .transform(Transformation::Orthographic(-1.0, 1.0, -1.0, 1.0, 0.0, -2.0)).start_t(8.5).end_t(9.5)
            // .transform(Transformation::Orthographic(-1.0, 1.0, -1.0, 1.0, 0.0, -1.0)).start_t(5.5).end_t(6.5)
            // .transform(Transformation::Orthographic(-1.0, 1.0, -1.0, 1.0, -10.0, 10.0)).start_t(0.0).end_t(0.0)
            ,
    }
}
