use std::f32::consts::PI;

use vera::*;
use vera_core::*;

fn main() {
    let mut v = Vera::init(get());
    unsafe {
        D_VERTEX_ALPHA = 1.0;
    }

    v.show();
}

fn get() -> Input {
    let points: u32 = 100;


    Input {
        meta: MetaInput {
            bg: [0.0, 0.0, 0.0, 1.0],
            start: -2.0,
            end: 7.0,
        },
        m: {
            let text = text::Text::new("Circle".to_owned(), 0.1, 0.2, 0.0).model()
                .rgb(1.0, 1.0, 1.0)
                .alpha(1.0)
                .transform(Transformation::Scale(1.0, -1.0, 1.0)).start_t(-2.0).end_t(-2.0)
                .transform(Transformation::Translate(-0.2, 0.0, 0.0)).start_t(-2.0).end_t(-2.0)
                .recolor(Colorization::ToColor(1.0, 1.0, 1.0, 0.0))
                .start_c(-1.0)
                .end_c(0.5);
            let mut points: Vec<Model> = (0..points)
                .map(|n| {
                    let a = n as f32 / points as f32 * 2.0;
                    point(0.9, 0.0, 0.01, a)
                        .transform(Transformation::RotateZ(a * PI))
                        .evolution_t(Evolution::FastIn)
                        .start_t(2.0-a)
                        .end_t(4.0)
                }
                ).collect();
            points.push(text);
            points
        }, 
        v: View::new(),
        p: Projection::new(),
    }
}

fn point(x: f32, y: f32, rad: f32, spawn: f32) -> Model {
    circle(x, y, rad, spawn)
        .rgb(1.0, 1.0, 1.0)
        .alpha(0.0)
}

fn circle(x: f32, y: f32, rad: f32, spawn: f32) -> Model {
    Model::from_models(
        (0..100).map(|n| Model::from_vertices(vec![
            Vertex::new().pos(x, y, 0.0),
            Vertex::new().pos(x + (  n   as f32 / 50.0 * PI).cos() * rad, y + (  n   as f32 / 50.0 * PI).sin() * rad, 0.0),
            Vertex::new().pos(x + ((n+1) as f32 / 50.0 * PI).cos() * rad, y + ((n+1) as f32 / 50.0 * PI).sin() * rad, 0.0),
        ])
            .recolor(Colorization::ToColor(1.0, 1.0, 1.0, 1.0))
            .start_c(2.0-spawn)
            .end_c(4.0-spawn)
            .evolution_c(Evolution::Linear))
        .collect()
    )
}