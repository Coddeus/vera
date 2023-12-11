use vera_shapes::{Shape, Triangle, Vertex};

#[no_mangle]
pub fn get() -> Vec<Shape> {
    // unsafe {
    //     vera_shapes::D_RANDOM_COLORS = true;
    // }

    let t1 = Triangle::new(
        Vertex::new().pos(0.0, 0.5, 0.0),
        Vertex::new().pos(0.3, 0.2, 0.0),
        Vertex::new().pos(-0.2, 0.4, 0.0),
    );

    
    vec![
        Shape::from_vertices(vec![
            Vertex::new().pos(0.0, 0.0, 0.0),
            Vertex::new().pos(0.5, 0.5, 0.5),
            Vertex::new().pos(0.0, 1.0, 0.0),
        ]),
        t1,
    ]
}