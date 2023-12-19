use std::f32::consts::PI;

use vera::*;
use vera_shapes::*;

fn main() {
    let mut v = Vera::create(get());
    unsafe {
        D_VERTEX_ALPHA = 1.0;
    }

    while v.show() {
        v.reset(get());
    };
}

fn get() -> Input {
    Input {
        meta: MetaInput {
            bg: [0.1, 0.3, 0.5, 1.0],
            start: 0.0,
            end: 13.0,
        },
        m: (0..100)
            .map(|n| 
                Model::from_vertices(
                    vec![
                        Vertex::new().pos(-0.5, -0.5, 0.0).rgba((99-n) as f32/99.0, 0.0, 0.0, 1.0),
                        Vertex::new().pos(0.5, -0.5, 0.0).rgba((99-n) as f32/99.0, 0.0, 0.0, 1.0),
                        Vertex::new().pos(-0.5, 0.5, 0.0).rgba((99-n) as f32/99.0, 0.0, 0.0, 1.0),

                        Vertex::new().pos(0.5, 0.5, 0.0).rgba((99-n) as f32/99.0, 0.0, 0.0, 1.0),
                        Vertex::new().pos(0.5, -0.5, 0.0).rgba((99-n) as f32/99.0, 0.0, 0.0, 1.0),
                        Vertex::new().pos(-0.5, 0.5, 0.0).rgba((99-n) as f32/99.0, 0.0, 0.0, 1.0),
                    ]
                )
                .transform(Transformation::Scale(0.1, 0.1, 0.1)).start_t(1.0 + (100 - n) as f32 / 50.0).end_t(1.5 + (100 - n) as f32 / 50.0)
                .transform(Transformation::Scale(2.0, 2.0, 2.0)).start_t(9.0).end_t(11.0)
                .transform(Transformation::RotateZ(2.0 * PI)).start_t(n as f32/100.0 + 5.0).end_t(n as f32/100.0 + 7.0)
                .transform(Transformation::Translate(((n%10) as f32 - 4.5) / 5.0, ((n/10) as f32 - 4.5) / 5.0, 0.0)).start_t(1.0 + (100 - n) as f32 / 50.0).end_t(1.5 + (100 - n) as f32 / 50.0)
            ).collect(), 
        v: View::new(),
        p: Projection::new(),
    }
}

