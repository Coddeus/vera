use std::f32::consts::PI;

use vera_core::Vera;
use vera::{text::Text, Colorization, Evolution, Input, MetaInput, Model, Projection, ToModel, Transformation, Vertex, View};

fn main() {
    let mut v = Vera::init(Input {
        meta: MetaInput {
            bg: [0.1, 0.1, 0.1, 1.0],
            start: 0.0,
            end: 10.0,
        },
        m: {
            let small = Text::new("Small".to_owned(), 0.1, 0.0, 1.0);
            let big = Text::new("Big".to_owned(), 1.0, 0.0, 1.0);
            let hellotext = Text::new("Hello Colored Text!".to_owned(), 1.0, 0.0, 1.0);
            let iamspawning: Text = Text::new("I am spawning...".to_owned(), 1.0, 0.0, 1.0);
            let close: Text = Text::new("I am close".to_owned(), 0.8, -0.1, 1.0);
            let far: Text = Text::new("I am far".to_owned(), 0.8, 0.1, 1.0);
            let spawningtext = iamspawning.model().alpha(0.0);
            vec![
                small.model()
                    .rgb(1.0, 1.0, 1.0)
                    .transform(Transformation::Scale(1.0, -1.0, 1.0)).start_t(0.0).end_t(0.0)
                    .transform(Transformation::Translate(0.0, -0.5, 0.0)).start_t(0.0).end_t(0.0)
                    .transform(Transformation::Translate(-1.0, 0.0, 0.0)).start_t(5.0).end_t(7.0),
                big.model()
                    .rgb(1.0, 1.0, 1.0)
                    .transform(Transformation::Scale(1.0, -1.0, 1.0)).start_t(0.0).end_t(0.0)
                    .transform(Transformation::Translate(-0.2, -5.5, 0.0)).start_t(0.0).end_t(0.0)
                    .transform(Transformation::Translate(0.0, 5.2, 0.0)).start_t(6.8).end_t(7.0),
                hellotext.model()
                    .transform(Transformation::Scale(0.25, -0.25, 0.25)).start_t(0.0).end_t(0.0)
                    .transform(Transformation::Translate(-1.1, 0.0, 0.0)).start_t(0.0).end_t(0.0),
                close.model()
                    .transform(Transformation::Scale(0.25, -0.25, 0.25)).start_t(0.0).end_t(0.0)
                    .transform(Transformation::Translate(-0.9, 0.4, 0.0)).start_t(0.0).end_t(0.0),
                far.model()
                    .transform(Transformation::Scale(0.25, -0.25, 0.25)).start_t(0.0).end_t(0.0)
                    .transform(Transformation::Translate(-0.1, 0.4, 0.0)).start_t(0.0).end_t(0.0),
                spawningtext
                    .transform(Transformation::Scale(0.25, -0.25, 0.25)).start_t(0.0).end_t(0.0)
                    .transform(Transformation::Translate(-0.7, 0.7, 0.0)).start_t(0.0).end_t(0.0),
            ]
        },
        v: View::new(),
        p: Projection::new(),
    });
    v.show();
}