use vera::*;

#[hot_lib_reloader::hot_module(dylib = "lib")]
mod hot_lib {
    use vera_shapes::{Model, Input};
    // Path form the project root

    #[hot_function]
    pub fn get() -> Input {}
}

fn main() {
    let mut v = Vera::create(hot_lib::get());

    'dev: loop {
        if !v.show() {
            break 'dev;
        }
    }
}
