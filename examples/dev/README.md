# Dev
Vera with hot-reload for development.

## Usage
In the folder of this README:
```shell
cargo watch -w lib -x 'build -p lib'
# Another terminal
cargo run
```

Then, modifying the get() function in `lib/src/lib.rs` will modify the render when the animation restarts.  
src/main.rs is an intermediate. It sends the `Input` returned from the hot lib to vera.