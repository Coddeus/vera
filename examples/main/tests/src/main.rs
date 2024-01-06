use vera::D_VERTEX_ALPHA;
use vera_core::Vera;

mod anim;

fn main() {
    println!("Running tests.");
    println!("1. Static drawing,");
    println!("2. Per-vertex animation,");
    println!("3. Per-model animation,");
    unsafe {
        D_VERTEX_ALPHA = 1.0;
    }

    let mut v = Vera::init(anim::get());
    v.show();
}
