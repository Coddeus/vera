mod triangle;
mod vertex;
mod viewport;

pub use triangle::*;
pub use vertex::*;
pub use viewport::*;

pub struct Element {
    id: u32,
    vert_offset: u32,
    vert_len: u32,
    // anchor: Anchor,
}