# Vera-shapes
Defines shapes used by [the Vera Core](https://docs.rs/vera).

They are defined independently of vulkano, for faster hot-reload.
They are not used directly for the graphics pipeline (e.g. vertex/uniform input), but contain everything for the Vera core to recreate the shape, and update it with time.