use vera::*;

// Tilings / Tessellations.

#[no_mangle]
fn get() -> Input {
    unsafe {
        D_TRANSFORMATION_START_TIME = -1.;
        D_TRANSFORMATION_END_TIME = -1.;
        D_COLORIZATION_START_TIME = -1.;
        D_COLORIZATION_END_TIME = -1.;
    }
    Input {
        meta: MetaInput {
            bg: [0.1, 0.1, 0.1, 0.1],
            start: 0.0,
            end: 10.0,
        },
        m: vec![
            Model::from_vertices(vec![
                Vertex::new().pos(0., 1., 0.).b(1.),
                Vertex::new().pos(3.0f32.sqrt()/2., -0.5, 0.).b(1.),
                Vertex::new().pos(-3.0f32.sqrt()/2., -0.5, 0.).b(1.),
            ])
        ],
        v: View::new().transform(Transformation::Lookat(0., 0., -3., 0., 0., 0., 0., -1., 0.)),
        p: Projection::new().transform(Transformation::Perspective(-0.1, 0.1, -0.1, 0.1, 0.2, 100.)),
    }
}