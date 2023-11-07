use vera::*;

#[hot_lib_reloader::hot_module(
    dylib = "lib"
)]
mod hot_lib {
    use vera::shape::*;
    hot_functions_from_file!("lib/src/lib.rs");
}

fn main() {
    let mut v = Vera::create(1_000_000, 10_000, hot_lib::get());

    'dev: loop {
        match v.vk.show(&mut v.event_loop, (0, 0)) {
            0 => { // Successfully finished
                // Reset input data
                v.reset(hot_lib::get());
            }
            1 => { // Window closed 
                println!("\nℹ Window closed. Exiting.");
                break 'dev;
            }
            _ => {
                panic!("🛑 Unexpected return code when running the main loop");
            }
        }
    }
}
