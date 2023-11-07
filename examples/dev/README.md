### *Hot-lib-reloader*
*Files generated with `cargo generate rksm/rust-hot-reload`.*  
*This is an example workflow to use hot-lib-reloader while coding. See https://github.com/rksm/hot-lib-reloader-rs and https://robert.kra.hn/posts/hot-reloading-rust/ for more info.*

## Usage

In the folder of this README:
```shell
cargo watch -w lib -x 'build -p lib'
# Another terminal
cargo run
```