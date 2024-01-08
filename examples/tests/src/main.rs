use vera::D_VERTEX_ALPHA;
use vera_core::*;
use vera::{Input, MetaInput, Model, Vertex, Transformation, View, Projection, Colorization};

use std::f32::consts::PI;
#[hot_lib_reloader::hot_module(dylib = "anim")]
mod hot_lib {
    use vera::Input;
    // Path form the project root

    #[hot_function]
    pub fn get() -> Input {}
}

fn main() {
    println!("Running tests.");
    println!("1. Static drawing,");
    println!("2. Per-vertex animation,");
    println!("3. Per-model animation,");
    unsafe {
        D_VERTEX_ALPHA = 1.0;
    }

    // Yeah, no point in having hot-reloading there ;)
    let mut v = Vera::init(hot_lib::get());
    v.show();
}
