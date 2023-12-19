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
        meta: Default::default(),
        m: vec![
            circle(-0.8, Evolution::FastIn),
            circle(-0.4, Evolution::FastInOut),
            circle(0.0, Evolution::Linear),
            circle(0.4, Evolution::FastMiddle),
            circle(0.8, Evolution::FastOut),
        ], 
        v: View::new(),
        p: Projection::new(),
    }
}

fn circle(height: f32, evolution: Evolution) -> Model {
    Model::from_merge(
        (0..100).map(|n| Model::from_vertices(vec![
            Vertex::new().pos(0.0, 0.0, 0.0),
            Vertex::new().pos((  n   as f32 / 50.0 * PI).cos() * 0.15, (  n   as f32 / 50.0 * PI).sin() * 0.15, 0.0),
            Vertex::new().pos(((n+1) as f32 / 50.0 * PI).cos() * 0.15, ((n+1) as f32 / 50.0 * PI).sin() * 0.15, 0.0),
        ])).collect()
    )
        .rgb(1.0, 0.0, 0.0)
        .transform(Transformation::Translate(-0.75, height, 0.0)).start_t(0.0).end_t(0.0)
        .transform(Transformation::Translate(1.5, 0.0, 0.0)).start_t(1.0).end_t(2.0).evolution_t(evolution)
        .recolor(Colorization::ToColor(0.0, 0.0, 1.0, 1.0)).start_c(1.0).end_c(2.0).evolution_c(evolution)
}