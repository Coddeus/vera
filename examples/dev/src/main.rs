use vera::*;

#[hot_lib_reloader::hot_module(dylib = "lib")]
mod hot_lib {
    use vera_shapes::Model;
    // Path form the project root

    #[hot_function]
    pub fn get() -> Vec<Model> {}
}

fn main() {
    let mut v = Vera::create(hot_lib::get());

    'dev: loop {
        match v.vk.show(&mut v.event_loop, (0, 0)) {
            0 => {
                // Successfully finished
                // Reset input data
                v.reset(hot_lib::get());
            }
            1 => {
                // Window closed
                println!("\nℹ Window closed. Exiting.");
                break 'dev;
            }
            _ => {
                panic!("🛑 Unexpected return code when running the main loop");
            }
        }
    }
}
