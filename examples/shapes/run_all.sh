#!/bin/bash

# Loop through each directory in the current folder
for dir in */; do
    # Check if the directory contains a Cargo.toml file (Rust project)
    if [ -f "${dir}Cargo.toml" ]; then
        # Change directory and run `cargo run`
        echo "Running cargo run in ${dir}..."
        cd "$dir" || exit
        cargo run
        cd ..  # Go back to the original directory
    fi
done