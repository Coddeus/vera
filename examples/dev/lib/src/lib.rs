use std::f32::consts::PI;

use fastrand::f32;
use vera::{Input, MetaInput, View, Projection, Transformation, Model, Vertex, D_TRANSFORMATION_START_TIME, D_TRANSFORMATION_END_TIME};

const PHI: f32 = 1.618033988749;


#[no_mangle]
fn get() -> Input {
    unsafe {
        D_TRANSFORMATION_START_TIME = 0.;
        D_TRANSFORMATION_END_TIME = 0.;
    }
    Input {
        meta: MetaInput {
            bg: [0.3, 0.3, 0.3, 0.3],
            start: 0.0,
            end: 3.0,
        },
        m: vec![
            isocahedron()
        ],
        v: View::new().transform(Transformation::Lookat(0., 0., -6., 0., 0., 0., 0., 1., 0.)),
        p: Projection::new().transform(Transformation::Perspective(-0.1, 0.1, -0.1, 0.1, 0.2, 100.)),
    }
}

// fn sphere() -> Model {
// 
// }

/// The base isocahedron
fn isocahedron() -> Model {
    Model::from_vertices(vec![
        Vertex::new().pos(0., 1., PHI).rgba(f32(), 0., 0.2, 0.9)      ,Vertex::new().pos(1., PHI, 0.).rgba(f32(), 0., 0.2, 0.9)     ,Vertex::new().pos(-1., PHI, 0.).rgba(f32(), 0., 0.2, 0.9)     ,
        Vertex::new().pos(0., 1., PHI).rgba(f32(), 0., 0.2, 0.9)      ,Vertex::new().pos(-1., PHI, 0.).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(-PHI, 0., 1.).rgba(f32(), 0., 0.2, 0.9)     ,
        Vertex::new().pos(0., 1., PHI).rgba(f32(), 0., 0.2, 0.9)      ,Vertex::new().pos(-PHI, 0., 1.).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(0., -1., PHI).rgba(f32(), 0., 0.2, 0.9)     ,
        Vertex::new().pos(0., 1., PHI).rgba(f32(), 0., 0.2, 0.9)      ,Vertex::new().pos(0., -1., PHI).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(PHI, 0., 1.).rgba(f32(), 0., 0.2, 0.9)      ,
        Vertex::new().pos(0., 1., PHI).rgba(f32(), 0., 0.2, 0.9)      ,Vertex::new().pos(PHI, 0., 1.).rgba(f32(), 0., 0.2, 0.9)     ,Vertex::new().pos(1., PHI, 0.).rgba(f32(), 0., 0.2, 0.9)      ,

        Vertex::new().pos(0., -1., PHI).rgba(f32(), 0., 0.2, 0.9)     ,Vertex::new().pos(-PHI, 0., 1.).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(-1., -PHI, 0.).rgba(f32(), 0., 0.2, 0.9)    ,
        Vertex::new().pos(0., -1., PHI).rgba(f32(), 0., 0.2, 0.9)     ,Vertex::new().pos(-1., -PHI, 0.).rgba(f32(), 0., 0.2, 0.9)   ,Vertex::new().pos(1., -PHI, 0.).rgba(f32(), 0., 0.2, 0.9)     ,
        Vertex::new().pos(0., -1., PHI).rgba(f32(), 0., 0.2, 0.9)     ,Vertex::new().pos(1., -PHI, 0.).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(PHI, 0., 1.).rgba(f32(), 0., 0.2, 0.9)      ,
        Vertex::new().pos(0., -1., -PHI).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(0., 1., -PHI).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(PHI, 0., -1.).rgba(f32(), 0., 0.2, 0.9)     ,
        Vertex::new().pos(0., -1., -PHI).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(PHI, 0., -1.).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(1., -PHI, 0.).rgba(f32(), 0., 0.2, 0.9)     ,

        Vertex::new().pos(0., -1., -PHI).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(1., -PHI, 0.).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(-1., -PHI, 0.).rgba(f32(), 0., 0.2, 0.9)    ,
        Vertex::new().pos(0., -1., -PHI).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(-1., -PHI, 0.).rgba(f32(), 0., 0.2, 0.9)   ,Vertex::new().pos(-PHI, 0., -1.).rgba(f32(), 0., 0.2, 0.9)    ,
        Vertex::new().pos(0., -1., -PHI).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(-PHI, 0., -1.).rgba(f32(), 0., 0.2, 0.9)   ,Vertex::new().pos(0., 1., -PHI).rgba(f32(), 0., 0.2, 0.9)     ,
        Vertex::new().pos(0., 1., -PHI).rgba(f32(), 0., 0.2, 0.9)     ,Vertex::new().pos(-PHI, 0., -1.).rgba(f32(), 0., 0.2, 0.9)   ,Vertex::new().pos(-1., PHI, 0.).rgba(f32(), 0., 0.2, 0.9)     ,
        Vertex::new().pos(0., 1., -PHI).rgba(f32(), 0., 0.2, 0.9)     ,Vertex::new().pos(-1., PHI, 0.).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(1., PHI, 0.).rgba(f32(), 0., 0.2, 0.9)      ,

        Vertex::new().pos(0., 1., -PHI).rgba(f32(), 0., 0.2, 0.9)     ,Vertex::new().pos(1., PHI, 0.).rgba(f32(), 0., 0.2, 0.9)     ,Vertex::new().pos(PHI, 0., -1.).rgba(f32(), 0., 0.2, 0.9)     ,
        Vertex::new().pos(PHI, 0., 1.).rgba(f32(), 0., 0.2, 0.9)      ,Vertex::new().pos(1., -PHI, 0.).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(PHI, 0., -1.).rgba(f32(), 0., 0.2, 0.9)     ,
        Vertex::new().pos(PHI, 0., 1.).rgba(f32(), 0., 0.2, 0.9)      ,Vertex::new().pos(PHI, 0., -1.).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(1., PHI, 0.).rgba(f32(), 0., 0.2, 0.9)      ,
        Vertex::new().pos(-PHI, 0., 1.).rgba(f32(), 0., 0.2, 0.9)     ,Vertex::new().pos(-1., PHI, 0.).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(-PHI, 0., -1.).rgba(f32(), 0., 0.2, 0.9)    ,
        Vertex::new().pos(-PHI, 0., 1.).rgba(f32(), 0., 0.2, 0.9)     ,Vertex::new().pos(-PHI, 0., -1.).rgba(f32(), 0., 0.2, 0.9)   ,Vertex::new().pos(-1., -PHI, 0.).rgba(f32(), 0., 0.2, 0.9)    ,
    ])
    .transform(Transformation::RotateX(PI/3.))
    .transform(Transformation::RotateY(PI/3.))
    .transform(Transformation::RotateY(2.*PI)).start_t(0.0).end_t(3.0)
}