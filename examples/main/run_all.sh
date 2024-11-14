#!/bin/bash

for dir in */; do
    if [ -f "${dir}Cargo.toml" ]; then
        echo "Running cargo run in ${dir}..."
        cd "$dir" || exit
        cargo run
        cd ..
    fi
done