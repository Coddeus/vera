use std::f32::consts::PI;
use itertools::Itertools;

use vera_core::Vera;
use vera::{Colorization, Evolution, Input, MetaInput, Model, Projection, Transformation, Vertex, View};

fn main() {
    let mut v = Vera::init(Input {
        meta: MetaInput {
            bg: [0.1, 0.1, 0.1, 1.0],
            start: 0.0,
            end: 13.0,
        },
        m: (0..2).flat_map(move |rot| {
            (-9..10).flat_map(move |col| {
                (-9..10).map(move |row| {
                    Model::from_vertices(vec![
                        Vertex::new().pos(0., 1., 0.).b(1.).recolor(Colorization::ToColor(0., 0., 0., 0.)).start_c(11.0 + (10*row + col) as f32 / 100.0).end_c(11.5 + (10*row + col) as f32 / 100.0).evolution_c(Evolution::SlowIn),
                        Vertex::new().pos(3.0f32.sqrt()/2., -0.5, 0.).b(1.).recolor(Colorization::ToColor(0., 0., 0., 0.)).start_c(11.0 + (10*row + col) as f32 / 100.0).end_c(11.5 + (10*row + col) as f32 / 100.0).evolution_c(Evolution::SlowIn),
                        Vertex::new().pos(-3.0f32.sqrt()/2., -0.5, 0.).b(1.).recolor(Colorization::ToColor(0., 0., 0., 0.)).start_c(11.0 + (10*row + col) as f32 / 100.0).end_c(11.5 + (10*row + col) as f32 / 100.0).evolution_c(Evolution::SlowIn),
                    ])
                        .transform(Transformation::Scale(0.001, 0.001, 0.001)).start_t(0.0).end_t(0.0)
                        .transform(Transformation::Scale(1000.0, 1000.0, 1000.0)).start_t(0.0).end_t(1.5).evolution_t(Evolution::FastIn)
                        .transform(Transformation::RotateZ(rot as f32 * PI / 3.)).start_t(5.5 - 4.0 * ((col as f32).abs() + (row as f32).abs())/20.).end_t(8.0 - 4.0 * ((col as f32).abs() + (row as f32).abs())/20.).evolution_t(Evolution::FastOut)
                        .transform(Transformation::Translate(3.0f32.sqrt() * rot as f32 / 2.0, 0.5 * rot as f32, 0.0)).start_t(5.5 - 4.0 * ((col as f32).abs() + (row as f32).abs())/20.).end_t(8.0 - 4.0 * ((col as f32).abs() + (row as f32).abs())/20.).evolution_t(Evolution::FastOut)
                        .transform(Transformation::Translate(col as f32 * 3.0f32.sqrt(), row as f32 * 1.5, 0.0)).start_t(5.5 - 4.0 * ((col as f32).abs() + (row as f32).abs())/20.).end_t(8.0 - 4.0 * ((col as f32).abs() + (row as f32).abs())/20.).evolution_t(Evolution::FastOut)
                })
            })
        }).collect_vec(),
        v: View::new().transform(Transformation::Lookat(0., 0., -3., 0., 0., 0., 0., 1., 0.)).start_t(0.).end_t(0.),
        p: Projection::new().transform(Transformation::Perspective(-0.1, 0.1, -0.1, 0.1, 0.1, 100.)).start_t(0.).end_t(0.),
    });
    v.show();
}