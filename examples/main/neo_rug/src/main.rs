//! This example was made before animation was implemented. It calculates vertices directly for a static image.

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
    Input {
        meta: Default::default(),
        m: (0..100)
        .into_iter()
        .map(|n| {
            Model::from_models({
                vec![
                    Triangle::new(
                        Vertex::new()
                            .pos((n / 10 - 5) as f32 / 5.0, (n % 10 - 5) as f32 / 5.0, 0.0)
                            .rgb(0.0, 1.0, 1.0),
                        Vertex::new()
                            .pos(
                                ((n / 10 - 5) as f32 + 0.5) / 5.0,
                                (n % 10 - 5) as f32 / 5.0,
                                0.0,
                            )
                            .rgb(1.0, 0.0, 1.0),
                        Vertex::new()
                            .pos(
                                (n / 10 - 5) as f32 / 5.0,
                                ((n % 10 - 5) as f32 + 0.5) / 5.0,
                                0.0,
                            )
                            .rgb(1.0, 0.0, 1.0),
                    ),
                    Triangle::new(
                        Vertex::new()
                            .pos(
                                ((n / 10 - 5) as f32 + 0.5) / 5.0,
                                (n % 10 - 5) as f32 / 5.0,
                                0.0,
                            )
                            .rgb(1.0, 0.0, 1.0),
                        Vertex::new()
                            .pos(
                                (n / 10 - 5) as f32 / 5.0,
                                ((n % 10 - 5) as f32 + 0.5) / 5.0,
                                0.0,
                            )
                            .rgb(1.0, 0.0, 1.0),
                        Vertex::new()
                            .pos(
                                ((n / 10 - 5) as f32 + 0.5) / 5.0,
                                ((n % 10 - 5) as f32 + 0.5) / 5.0,
                                0.0,
                            )
                            .rgb(0.0, 1.0, 1.0),
                    ),
                    Triangle::new(
                        Vertex::new()
                            .pos(
                                ((n / 10 - 5) as f32 + 0.5) / 5.0,
                                ((n % 10 - 5) as f32 + 0.5) / 5.0,
                                0.0,
                            )
                            .rgb(0.0, 1.0, 1.0),
                        Vertex::new()
                            .pos(
                                ((n / 10 - 5) as f32 + 1.0) / 5.0,
                                ((n % 10 - 5) as f32 + 0.5) / 5.0,
                                0.0,
                            )
                            .rgb(1.0, 0.0, 1.0),
                        Vertex::new()
                            .pos(
                                ((n / 10 - 5) as f32 + 0.5) / 5.0,
                                ((n % 10 - 5) as f32 + 1.0) / 5.0,
                                0.0,
                            )
                            .rgb(1.0, 0.0, 1.0),
                    ),
                    Triangle::new(
                        Vertex::new()
                            .pos(
                                ((n / 10 - 5) as f32 + 1.0) / 5.0,
                                ((n % 10 - 5) as f32 + 0.5) / 5.0,
                                0.0,
                            )
                            .rgb(1.0, 0.0, 1.0),
                        Vertex::new()
                            .pos(
                                ((n / 10 - 5) as f32 + 0.5) / 5.0,
                                ((n % 10 - 5) as f32 + 1.0) / 5.0,
                                0.0,
                            )
                            .rgb(1.0, 0.0, 1.0),
                        Vertex::new()
                            .pos(
                                ((n / 10 - 5) as f32 + 1.0) / 5.0,
                                ((n % 10 - 5) as f32 + 1.0) / 5.0,
                                0.0,
                            )
                            .rgb(0.0, 1.0, 1.0),
                    ),
                    Triangle::new(
                        Vertex::new()
                            .pos(
                                ((n / 10 - 5) as f32 + 1.0) / 5.0,
                                (n % 10 - 5) as f32 / 5.0,
                                0.0,
                            )
                            .rgb(0.0, 1.0, 1.0),
                        Vertex::new()
                            .pos(
                                ((n / 10 - 5) as f32 + 1.0) / 5.0,
                                ((n % 10 - 5) as f32 + 0.5) / 5.0,
                                0.0,
                            )
                            .rgb(1.0, 0.0, 1.0),
                        Vertex::new()
                            .pos(
                                ((n / 10 - 5) as f32 + 0.5) / 5.0,
                                (n % 10 - 5) as f32 / 5.0,
                                0.0,
                            )
                            .rgb(1.0, 0.0, 1.0),
                    ),
                    Triangle::new(
                        Vertex::new()
                            .pos(
                                ((n / 10 - 5) as f32 + 1.0) / 5.0,
                                ((n % 10 - 5) as f32 + 0.5) / 5.0,
                                0.0,
                            )
                            .rgb(1.0, 0.0, 1.0),
                        Vertex::new()
                            .pos(
                                ((n / 10 - 5) as f32 + 0.5) / 5.0,
                                (n % 10 - 5) as f32 / 5.0,
                                0.0,
                            )
                            .rgb(1.0, 0.0, 1.0),
                        Vertex::new()
                            .pos(
                                ((n / 10 - 5) as f32 + 0.5) / 5.0,
                                ((n % 10 - 5) as f32 + 0.5) / 5.0,
                                0.0,
                            )
                            .rgb(0.0, 1.0, 1.0),
                    ),
                    Triangle::new(
                        Vertex::new()
                            .pos(
                                ((n / 10 - 5) as f32 + 0.5) / 5.0,
                                ((n % 10 - 5) as f32 + 0.5) / 5.0,
                                0.0,
                            )
                            .rgb(0.0, 1.0, 1.0),
                        Vertex::new()
                            .pos(
                                ((n / 10 - 5) as f32 + 0.5) / 5.0,
                                ((n % 10 - 5) as f32 + 1.0) / 5.0,
                                0.0,
                            )
                            .rgb(1.0, 0.0, 1.0),
                        Vertex::new()
                            .pos(
                                (n / 10 - 5) as f32 / 5.0,
                                ((n % 10 - 5) as f32 + 0.5) / 5.0,
                                0.0,
                            )
                            .rgb(1.0, 0.0, 1.0),
                    ),
                    Triangle::new(
                        Vertex::new()
                            .pos(
                                ((n / 10 - 5) as f32 + 0.5) / 5.0,
                                ((n % 10 - 5) as f32 + 1.0) / 5.0,
                                0.0,
                            )
                            .rgb(1.0, 0.0, 1.0),
                        Vertex::new()
                            .pos(
                                (n / 10 - 5) as f32 / 5.0,
                                ((n % 10 - 5) as f32 + 0.5) / 5.0,
                                0.0,
                            )
                            .rgb(1.0, 0.0, 1.0),
                        Vertex::new()
                            .pos(
                                (n / 10 - 5) as f32 / 5.0,
                                ((n % 10 - 5) as f32 + 1.0) / 5.0,
                                0.0,
                            )
                            .rgb(0.0, 1.0, 1.0),
                    ),
                ]
            })
        })
        .collect(), 
        v: View::new(),
        p: Projection::new(),
    }

    // // More triangles
    // (0..10000)
    //     .into_iter()
    //     .map(|n| Shape::from_merge({
    //         vec![
    //             Triangle::new(
    //                 Vertex::new().pos((n/100-50) as f32 / 50.0         , (n%100-50) as f32 / 50.0         , 0.0).rgb(0.0, 1.0, 1.0),
    //                 Vertex::new().pos(((n/100-50) as f32 + 0.5) / 50.0 , (n%100-50) as f32 / 50.0         , 0.0).rgb(1.0, 0.0, 1.0),
    //                 Vertex::new().pos((n/100-50) as f32 / 50.0         , ((n%100-50) as f32 + 0.5) / 50.0 , 0.0).rgb(1.0, 0.0, 1.0),
    //             ),
    //             Triangle::new(
    //                 Vertex::new().pos(((n/100-50) as f32 + 0.5) / 50.0 , (n%100-50) as f32 / 50.0         , 0.0).rgb(1.0, 0.0, 1.0),
    //                 Vertex::new().pos((n/100-50) as f32 / 50.0         , ((n%100-50) as f32 + 0.5) / 50.0 , 0.0).rgb(1.0, 0.0, 1.0),
    //                 Vertex::new().pos(((n/100-50) as f32 + 0.5) / 50.0 , ((n%100-50) as f32 + 0.5) / 50.0 , 0.0).rgb(0.0, 1.0, 1.0),
    //             ),
    //             Triangle::new(
    //                 Vertex::new().pos(((n/100-50) as f32 + 0.5) / 50.0 , ((n%100-50) as f32 + 0.5) / 50.0 , 0.0).rgb(0.0, 1.0, 1.0),
    //                 Vertex::new().pos(((n/100-50) as f32 + 1.0) / 50.0 , ((n%100-50) as f32 + 0.5) / 50.0 , 0.0).rgb(1.0, 0.0, 1.0),
    //                 Vertex::new().pos(((n/100-50) as f32 + 0.5) / 50.0 , ((n%100-50) as f32 + 1.0) / 50.0 , 0.0).rgb(1.0, 0.0, 1.0),
    //             ),
    //             Triangle::new(
    //                 Vertex::new().pos(((n/100-50) as f32 + 1.0) / 50.0 , ((n%100-50) as f32 + 0.5) / 50.0 , 0.0).rgb(1.0, 0.0, 1.0),
    //                 Vertex::new().pos(((n/100-50) as f32 + 0.5) / 50.0 , ((n%100-50) as f32 + 1.0) / 50.0 , 0.0).rgb(1.0, 0.0, 1.0),
    //                 Vertex::new().pos(((n/100-50) as f32 + 1.0) / 50.0 , ((n%100-50) as f32 + 1.0) / 50.0 , 0.0).rgb(0.0, 1.0, 1.0),
    //             ),
    //             Triangle::new(
    //                 Vertex::new().pos(((n/100-50) as f32 + 1.0) / 50.0 , (n%100-50) as f32 / 50.0         , 0.0).rgb(0.0, 1.0, 1.0),
    //                 Vertex::new().pos(((n/100-50) as f32 + 1.0) / 50.0 , ((n%100-50) as f32 + 0.5) / 50.0 , 0.0).rgb(1.0, 0.0, 1.0),
    //                 Vertex::new().pos(((n/100-50) as f32 + 0.5) / 50.0 , (n%100-50) as f32 / 50.0         , 0.0).rgb(1.0, 0.0, 1.0),
    //             ),
    //             Triangle::new(
    //                 Vertex::new().pos(((n/100-50) as f32 + 1.0) / 50.0 , ((n%100-50) as f32 + 0.5) / 50.0 , 0.0).rgb(1.0, 0.0, 1.0),
    //                 Vertex::new().pos(((n/100-50) as f32 + 0.5) / 50.0 , (n%100-50) as f32 / 50.0         , 0.0).rgb(1.0, 0.0, 1.0),
    //                 Vertex::new().pos(((n/100-50) as f32 + 0.5) / 50.0 , ((n%100-50) as f32 + 0.5) / 50.0 , 0.0).rgb(0.0, 1.0, 1.0),
    //             ),
    //             Triangle::new(
    //                 Vertex::new().pos(((n/100-50) as f32 + 0.5) / 50.0 , ((n%100-50) as f32 + 0.5) / 50.0 , 0.0).rgb(0.0, 1.0, 1.0),
    //                 Vertex::new().pos(((n/100-50) as f32 + 0.5) / 50.0 , ((n%100-50) as f32 + 1.0) / 50.0 , 0.0).rgb(1.0, 0.0, 1.0),
    //                 Vertex::new().pos((n/100-50) as f32 / 50.0         , ((n%100-50) as f32 + 0.5) / 50.0 , 0.0).rgb(1.0, 0.0, 1.0),
    //             ),
    //             Triangle::new(
    //                 Vertex::new().pos(((n/100-50) as f32 + 0.5) / 50.0 , ((n%100-50) as f32 + 1.0) / 50.0 , 0.0).rgb(1.0, 0.0, 1.0),
    //                 Vertex::new().pos((n/100-50) as f32 / 50.0         , ((n%100-50) as f32 + 0.5) / 50.0 , 0.0).rgb(1.0, 0.0, 1.0),
    //                 Vertex::new().pos((n/100-50) as f32 / 50.0         , ((n%100-50) as f32 + 1.0) / 50.0 , 0.0).rgb(0.0, 1.0, 1.0),
    //             ),
    //         ]
    //     }))
    //     .collect()
}
