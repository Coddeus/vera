use vera::shape::Veratex;

#[no_mangle]
pub fn get() -> Vec<Veratex> {
    vec![
        Veratex::new(0.0, 0.0, 0),
        Veratex::new(-1.0, 0.0, 0),
        Veratex::new(0.0, -1.0, 0),
        Veratex::new(0.0, 0.0, 0),
        Veratex::new(1.0, 0.0, 0),
        Veratex::new(0.0, 1.0, 0),
    ]
}