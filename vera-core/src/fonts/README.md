# How to use a font
- Create a font atlas with msdf-atlas-gen
    - json for texture data
        - goes in vera/src/extensions/text/fonts
        - change path in vera/src/extensions/text/font.rs
    - png for image atlas
        - goes in vera-core/src/fonts
        - you may change it to another color format (e.g. `ffmpeg -i cmunti_msdf_100_005.png -pix_fmt rgba cmunti_msdf_100_005_rgba.png`)
        - change path for texture creation in vera-core/src/lib.rs, as well as buffer size and image format (to match the new file colorspace)
- Modify the shader according to the mode generated