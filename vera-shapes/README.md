# Vera-shapes
Defines the Code User Interface sent to [the Vera Core](https://docs.rs/vera) via its `Input` struct.

It is defined independently of vulkano, for faster hot-reload.
It is not used directly for the graphics pipeline (e.g. vertex/uniform input), but contain everything for the Vera core to create/animate the shapes once sent.