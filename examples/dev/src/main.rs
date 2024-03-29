use vera_core::*;

#[hot_lib_reloader::hot_module(dylib = "lib")]
mod hot_lib {
    use vera::Input;
    // Path form the project root

    #[hot_function]
    pub fn get() -> Input {}
}

fn main() {
    let mut v = Vera::init(hot_lib::get());

    // v.show();
    while v.show() {
        v.reset(hot_lib::get())
    }
}
