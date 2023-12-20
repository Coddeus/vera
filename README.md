# Vera
Vulkan Engine in Rust for Animation.  
In early development.

Development video series: [YouTube](https://www.youtube.com/playlist?list=PLFBSAg3dVe4z5HxaZmOH0gaojQH4tLEgF)

This repository is split in 2 crates, enabling faster hot-reload.
- `vera` for the (heavy) core engine,
- `vera-shapes` for shapes and transformations (should be the only one imported in the hot-reloaded library),

### Already here
- Draw custom red triangles on a window with a black background ("*early development*")
- Hot-reloaded workflow (see ./examples/dev)

### Coming
- Many shapes
- Many transformations
- Much :)