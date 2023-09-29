extern crate vera;
use vera::*;

fn main() {
    vk::main();
    let triangle = RegularPolygon::new(3);

    elements.add(
        triangle
    )

    // render();
    /* or */
    // save()
}