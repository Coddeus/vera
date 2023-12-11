use vera_shapes::{Shape, Triangle, Vertex, D_RANDOM_VERTEX_COLOR, D_VERTEX_POSITION, D_VERTEX_COLOR, D_VERTEX_ALPHA};

#[no_mangle]
pub fn get() -> Vec<Shape> {
    unsafe {
        D_RANDOM_VERTEX_COLOR = false;
        D_VERTEX_POSITION = [0.0, 0.0, 0.0];
        D_VERTEX_COLOR = [1.0, 1.0, 1.0];
        D_VERTEX_ALPHA = 1.0;
    }

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