use vera_shapes::{Shape, Triangle};

#[no_mangle]
pub fn get() -> Vec<Shape> {
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