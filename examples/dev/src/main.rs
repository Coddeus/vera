use vera::*;

#[hot_lib_reloader::hot_module(
    dylib = "lib"
)]
mod hot_lib {
    use vera::shape::Veratex;
    hot_functions_from_file!("lib/src/lib.rs");
}

fn main() {
    let mut v = Vera::create();

    'dev: loop {
        match v.vk.show(&mut v.event_loop, false) {
            0 => { // Successfully finished
                v.data(hot_lib::get());
                // () => Repeat
            }
            1 => { // Window closed 
                println!("ℹ Window closed. Exiting.");
                break 'dev;
            }
            _ => {
                panic!("🛑 Unexpected return code when running the main loop");
            }
        }
    }
    v.dev();
}
