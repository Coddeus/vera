# Vera
Vulkan Engine in Rust for Animation.  
In early development.

Development video series: [YouTube](https://www.youtube.com/playlist?list=PLFBSAg3dVe4z5HxaZmOH0gaojQH4tLEgF)

This repository is split in 2 crates, enabling faster hot-reload.
- `vera` for the (heavy) core engine,
- `vera-shapes` for shapes and transformations (should be the only one imported in the hot-reloaded library),

## Features
- Draw anything out of triangles, by creating models from vertices, or by merging together several models you have already created.
- Send metadata for the background color, start time, and end time of the animation.
---
- Choose the default color and position of vertices
- Modify the color and position of each vertex, independently.
- Modify the color and position of each model.
- Modify the camera view to look wherever you want.
- Modify the projection to any custom perspective.
---
- Choose the start time and end time of each modification.
- Every modification is done at runtime, but you can make them start and end both at 0.0 to apply it directly.  
- Here are the currently available transformations:

| Type of transformations | Available transformations                                                                 |
|-------------------------|-------------------------------------------------------------------------------------------|
| Vertex / Model          | <ul><li>Scale</li><li>RotateX</li><li>RotateY</li><li>RotateZ</li><li>Translate</li></ul> |
| View (= Camera)         | <ul><li>Lookat</li></ul>                                                                  |
| Projection              | <ul><li>Perspective</ul>                                                                  |
---
- Hot-reloaded workflow (see ./examples/dev).
---
- Examples to get what's possible to do, and inspire you to do something great.

#### Coming
- Much :)