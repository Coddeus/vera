use crate::{Veratex, Vera};

pub struct Triangle([Veratex ; 3]);

impl Triangle {
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32, x3: f32, y3: f32) -> Self {
        Triangle([Veratex::new(x1, y1, 0), Veratex::new(x2, y2, 0), Veratex::new(x3, y3, 0)])
    }
}

impl crate::Shape for Triangle {
    fn vertices(self) -> [Veratex ; 3] {
        self.0
    }
}

impl Vera {
    // pub fn triangle()
}