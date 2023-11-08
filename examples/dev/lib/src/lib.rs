use vera_shapes::*;

pub fn get<V: Vertex>() -> Vec<Shape<V>> {
    vec![
        Triangle::new(
            0.5, 0.5,
            -0.5, -0.5,
            1.0, 0.0
        ),
        Triangle::new(
            -1.0, -1.0,
            -1.0, 1.0,
            1.0, -1.0
        ),
        Triangle::new(
            0.0, 0.5,
            0.5, -0.5,
            1.0, 0.0
        ),
    ]
}