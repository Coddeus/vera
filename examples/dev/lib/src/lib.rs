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
                        ]).rgb((fastrand::f32()+1.0) / 2.0, 0.0, 0.0),
                        Model::from_vertices(vec![
                            Vertex::new().pos(1.0, 1.0, 1.0),
                            Vertex::new().pos(-1.0, 1.0, -1.0),
                            Vertex::new().pos(-1.0, -1.0, 1.0),
                        ]).rgb(0.0, (fastrand::f32()+1.0) / 2.0, 0.0),
                        Model::from_vertices(vec![
                            Vertex::new().pos(-1.0, -1.0, 1.0),
                            Vertex::new().pos(-1.0, 1.0, -1.0),
                            Vertex::new().pos(1.0, -1.0, -1.0),
                        ]).rgb((fastrand::f32()+1.0) / 2.0, 0.0, (fastrand::f32()+1.0) / 2.0).alpha(fastrand::f32() / 2.0),
                    ]
                )
                .transform(Transformation::Scale(0.1, 0.1, 0.1)).start_t(1.0+n as f32/250.0).end_t(1.5+n as f32/250.0).evolution_t(Evolution::FastIn)
                .transform(Transformation::Translate(0.8, 1.2, 0.8)).start_t(1.0+n as f32/250.0).end_t(1.5+n as f32/250.0).evolution_t(Evolution::FastIn)
                .transform(Transformation::Translate(-0.8, -1.2, -0.8)).start_t(2.0+n as f32/250.0).end_t(2.5+n as f32/250.0).evolution_t(Evolution::FastIn)
                .transform(Transformation::Translate(((n/16) as f32 - 1.5) / 2.0, (((n%16)/4) as f32 - 1.5) / 2.0, ((n%4) as f32 - 1.5) / 2.0)).start_t(2.0+n as f32/250.0).end_t(2.5+n as f32/250.0).evolution_t(Evolution::FastIn)
            ).collect(),
        v: View::new()
            .transform(Transformation::Lookat(3.0, 4.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)).start_t(0.0).end_t(0.0).evolution_t(Evolution::Linear)
            .transform(Transformation::Lookat(1.5, 2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)).start_t(1.0).end_t(2.0).evolution_t(Evolution::FastIn)
            .transform(Transformation::Lookat(0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)).start_t(3.0).end_t(5.0).evolution_t(Evolution::FastMiddle)
            .transform(Transformation::Lookat(0.0, 0.0, -21.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)).start_t(6.0).end_t(7.0).evolution_t(Evolution::FastMiddle)
            .transform(Transformation::Lookat(0.0, 0.0, -1.1, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)).start_t(7.0).end_t(9.0).evolution_t(Evolution::FastMiddle),
        p: Projection::new()
            .transform(Transformation::Perspective(-0.1, 0.1, -0.1, 0.1, 0.2, 25.0)).start_t(0.0).end_t(0.0).evolution_t(Evolution::Linear)
            .transform(Transformation::Perspective(-0.1, 0.1, -0.1, 0.1, 2.0, 25.0)).start_t(6.0).end_t(7.0).evolution_t(Evolution::FastMiddle)
            .transform(Transformation::Perspective(-0.1, 0.1, -0.1, 0.1, 0.02, 25.0)).start_t(7.0).end_t(9.0).evolution_t(Evolution::FastMiddle),
    }
}
