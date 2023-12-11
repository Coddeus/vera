use vera::*;
use vera_shapes::*;

fn main() {
    let mut v = Vera::create(get());

    'dev: loop {
        match v.vk.show(&mut v.event_loop, (0, 0)) {
            0 => { // Successfully finished
                // Reset input data
                v.reset(get());
            }
            1 => { // Window closed 
                println!("\nℹ Window closed. Exiting.");
                break 'dev;
            }
            _ => {
                panic!("🛑 Unexpected return code when running the main loop");
            }
        }
    }
}


fn get() -> Vec<Shape> {
    (0..100)
        .into_iter()
        .map(|n| Shape::from_merge({
            vec![
                Triangle::new(
                    Vertex::new().pos((n/10-5) as f32 / 5.0         , (n%10-5) as f32 / 5.0         , 0.0).color(0.0, 1.0, 1.0),
                    Vertex::new().pos(((n/10-5) as f32 + 0.5) / 5.0 , (n%10-5) as f32 / 5.0         , 0.0).color(1.0, 0.0, 1.0),
                    Vertex::new().pos((n/10-5) as f32 / 5.0         , ((n%10-5) as f32 + 0.5) / 5.0 , 0.0).color(1.0, 0.0, 1.0),
                ),
                Triangle::new(
                    Vertex::new().pos(((n/10-5) as f32 + 0.5) / 5.0 , (n%10-5) as f32 / 5.0         , 0.0).color(1.0, 0.0, 1.0),
                    Vertex::new().pos((n/10-5) as f32 / 5.0         , ((n%10-5) as f32 + 0.5) / 5.0 , 0.0).color(1.0, 0.0, 1.0),
                    Vertex::new().pos(((n/10-5) as f32 + 0.5) / 5.0 , ((n%10-5) as f32 + 0.5) / 5.0 , 0.0).color(0.0, 1.0, 1.0),
                ),
                Triangle::new(
                    Vertex::new().pos(((n/10-5) as f32 + 0.5) / 5.0 , ((n%10-5) as f32 + 0.5) / 5.0 , 0.0).color(0.0, 1.0, 1.0),
                    Vertex::new().pos(((n/10-5) as f32 + 1.0) / 5.0 , ((n%10-5) as f32 + 0.5) / 5.0 , 0.0).color(1.0, 0.0, 1.0),
                    Vertex::new().pos(((n/10-5) as f32 + 0.5) / 5.0 , ((n%10-5) as f32 + 1.0) / 5.0 , 0.0).color(1.0, 0.0, 1.0),
                ),
                Triangle::new(
                    Vertex::new().pos(((n/10-5) as f32 + 1.0) / 5.0 , ((n%10-5) as f32 + 0.5) / 5.0 , 0.0).color(1.0, 0.0, 1.0),
                    Vertex::new().pos(((n/10-5) as f32 + 0.5) / 5.0 , ((n%10-5) as f32 + 1.0) / 5.0 , 0.0).color(1.0, 0.0, 1.0),
                    Vertex::new().pos(((n/10-5) as f32 + 1.0) / 5.0 , ((n%10-5) as f32 + 1.0) / 5.0 , 0.0).color(0.0, 1.0, 1.0),
                ),
                Triangle::new(
                    Vertex::new().pos(((n/10-5) as f32 + 1.0) / 5.0 , (n%10-5) as f32 / 5.0         , 0.0).color(0.0, 1.0, 1.0),
                    Vertex::new().pos(((n/10-5) as f32 + 1.0) / 5.0 , ((n%10-5) as f32 + 0.5) / 5.0 , 0.0).color(1.0, 0.0, 1.0),
                    Vertex::new().pos(((n/10-5) as f32 + 0.5) / 5.0 , (n%10-5) as f32 / 5.0         , 0.0).color(1.0, 0.0, 1.0),
                ),
                Triangle::new(
                    Vertex::new().pos(((n/10-5) as f32 + 1.0) / 5.0 , ((n%10-5) as f32 + 0.5) / 5.0 , 0.0).color(1.0, 0.0, 1.0),
                    Vertex::new().pos(((n/10-5) as f32 + 0.5) / 5.0 , (n%10-5) as f32 / 5.0         , 0.0).color(1.0, 0.0, 1.0),
                    Vertex::new().pos(((n/10-5) as f32 + 0.5) / 5.0 , ((n%10-5) as f32 + 0.5) / 5.0 , 0.0).color(0.0, 1.0, 1.0),
                ),
                Triangle::new(
                    Vertex::new().pos(((n/10-5) as f32 + 0.5) / 5.0 , ((n%10-5) as f32 + 0.5) / 5.0 , 0.0).color(0.0, 1.0, 1.0),
                    Vertex::new().pos(((n/10-5) as f32 + 0.5) / 5.0 , ((n%10-5) as f32 + 1.0) / 5.0 , 0.0).color(1.0, 0.0, 1.0),
                    Vertex::new().pos((n/10-5) as f32 / 5.0         , ((n%10-5) as f32 + 0.5) / 5.0 , 0.0).color(1.0, 0.0, 1.0),
                ),
                Triangle::new(
                    Vertex::new().pos(((n/10-5) as f32 + 0.5) / 5.0 , ((n%10-5) as f32 + 1.0) / 5.0 , 0.0).color(1.0, 0.0, 1.0),
                    Vertex::new().pos((n/10-5) as f32 / 5.0         , ((n%10-5) as f32 + 0.5) / 5.0 , 0.0).color(1.0, 0.0, 1.0),
                    Vertex::new().pos((n/10-5) as f32 / 5.0         , ((n%10-5) as f32 + 1.0) / 5.0 , 0.0).color(0.0, 1.0, 1.0),
                ),
            ]
        }))
        .collect()
        // // More triangles
        // (0..10000)
        //     .into_iter()
        //     .map(|n| Shape::from_merge({
        //         vec![
        //             Triangle::new(
        //                 Vertex::new().pos((n/100-50) as f32 / 50.0         , (n%100-50) as f32 / 50.0         , 0.0).color(0.0, 1.0, 1.0),
        //                 Vertex::new().pos(((n/100-50) as f32 + 0.5) / 50.0 , (n%100-50) as f32 / 50.0         , 0.0).color(1.0, 0.0, 1.0),
        //                 Vertex::new().pos((n/100-50) as f32 / 50.0         , ((n%100-50) as f32 + 0.5) / 50.0 , 0.0).color(1.0, 0.0, 1.0),
        //             ),
        //             Triangle::new(
        //                 Vertex::new().pos(((n/100-50) as f32 + 0.5) / 50.0 , (n%100-50) as f32 / 50.0         , 0.0).color(1.0, 0.0, 1.0),
        //                 Vertex::new().pos((n/100-50) as f32 / 50.0         , ((n%100-50) as f32 + 0.5) / 50.0 , 0.0).color(1.0, 0.0, 1.0),
        //                 Vertex::new().pos(((n/100-50) as f32 + 0.5) / 50.0 , ((n%100-50) as f32 + 0.5) / 50.0 , 0.0).color(0.0, 1.0, 1.0),
        //             ),
        //             Triangle::new(
        //                 Vertex::new().pos(((n/100-50) as f32 + 0.5) / 50.0 , ((n%100-50) as f32 + 0.5) / 50.0 , 0.0).color(0.0, 1.0, 1.0),
        //                 Vertex::new().pos(((n/100-50) as f32 + 1.0) / 50.0 , ((n%100-50) as f32 + 0.5) / 50.0 , 0.0).color(1.0, 0.0, 1.0),
        //                 Vertex::new().pos(((n/100-50) as f32 + 0.5) / 50.0 , ((n%100-50) as f32 + 1.0) / 50.0 , 0.0).color(1.0, 0.0, 1.0),
        //             ),
        //             Triangle::new(
        //                 Vertex::new().pos(((n/100-50) as f32 + 1.0) / 50.0 , ((n%100-50) as f32 + 0.5) / 50.0 , 0.0).color(1.0, 0.0, 1.0),
        //                 Vertex::new().pos(((n/100-50) as f32 + 0.5) / 50.0 , ((n%100-50) as f32 + 1.0) / 50.0 , 0.0).color(1.0, 0.0, 1.0),
        //                 Vertex::new().pos(((n/100-50) as f32 + 1.0) / 50.0 , ((n%100-50) as f32 + 1.0) / 50.0 , 0.0).color(0.0, 1.0, 1.0),
        //             ),
        //             Triangle::new(
        //                 Vertex::new().pos(((n/100-50) as f32 + 1.0) / 50.0 , (n%100-50) as f32 / 50.0         , 0.0).color(0.0, 1.0, 1.0),
        //                 Vertex::new().pos(((n/100-50) as f32 + 1.0) / 50.0 , ((n%100-50) as f32 + 0.5) / 50.0 , 0.0).color(1.0, 0.0, 1.0),
        //                 Vertex::new().pos(((n/100-50) as f32 + 0.5) / 50.0 , (n%100-50) as f32 / 50.0         , 0.0).color(1.0, 0.0, 1.0),
        //             ),
        //             Triangle::new(
        //                 Vertex::new().pos(((n/100-50) as f32 + 1.0) / 50.0 , ((n%100-50) as f32 + 0.5) / 50.0 , 0.0).color(1.0, 0.0, 1.0),
        //                 Vertex::new().pos(((n/100-50) as f32 + 0.5) / 50.0 , (n%100-50) as f32 / 50.0         , 0.0).color(1.0, 0.0, 1.0),
        //                 Vertex::new().pos(((n/100-50) as f32 + 0.5) / 50.0 , ((n%100-50) as f32 + 0.5) / 50.0 , 0.0).color(0.0, 1.0, 1.0),
        //             ),
        //             Triangle::new(
        //                 Vertex::new().pos(((n/100-50) as f32 + 0.5) / 50.0 , ((n%100-50) as f32 + 0.5) / 50.0 , 0.0).color(0.0, 1.0, 1.0),
        //                 Vertex::new().pos(((n/100-50) as f32 + 0.5) / 50.0 , ((n%100-50) as f32 + 1.0) / 50.0 , 0.0).color(1.0, 0.0, 1.0),
        //                 Vertex::new().pos((n/100-50) as f32 / 50.0         , ((n%100-50) as f32 + 0.5) / 50.0 , 0.0).color(1.0, 0.0, 1.0),
        //             ),
        //             Triangle::new(
        //                 Vertex::new().pos(((n/100-50) as f32 + 0.5) / 50.0 , ((n%100-50) as f32 + 1.0) / 50.0 , 0.0).color(1.0, 0.0, 1.0),
        //                 Vertex::new().pos((n/100-50) as f32 / 50.0         , ((n%100-50) as f32 + 0.5) / 50.0 , 0.0).color(1.0, 0.0, 1.0),
        //                 Vertex::new().pos((n/100-50) as f32 / 50.0         , ((n%100-50) as f32 + 1.0) / 50.0 , 0.0).color(0.0, 1.0, 1.0),
        //             ),
        //         ]
        //     }))
        //     .collect()
}