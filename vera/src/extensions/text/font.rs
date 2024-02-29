use serde::Deserialize;
use once_cell::sync::Lazy;


pub(crate) static CMUNTI: Lazy<FontData> = Lazy::new(||
    font_from(include_str!("./fonts/cmunti_msdf_100_005.json"))
);

fn font_from(serd_string: &str) -> FontData {
    serde_json::from_str(serd_string).unwrap()
}

pub(crate) fn char_bounds(c: u8) -> Option<(AtlasBounds, PlaneBounds, f64)> {
    let idx = match CMUNTI.glyphs.iter().position(|glyph| glyph.unicode == c as u32) {
        Some(idx) => idx,
        None => {
            return None
        }
    };
    Some( (CMUNTI.glyphs[idx].atlasBounds.unwrap(), CMUNTI.glyphs[idx].planeBounds.unwrap(), CMUNTI.glyphs[idx].advance) )
}
pub(crate) fn space_advance() -> Option<f64> {
    let space_unicode = ' ' as u32;
    let space_idx = match CMUNTI.glyphs.iter().position(|glyph| glyph.unicode == space_unicode) {
        Some(idx) => idx,
        None => {
            return None
        }
    };
    Some(CMUNTI.glyphs[space_idx].advance)
}


#[derive(Debug, Deserialize)]
#[allow(unused, non_snake_case)]
pub(crate) struct Atlas {
    #[serde(rename = "type")]
    atlas_type: String,
    distanceRange: u32,
    pub(crate) size: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    yOrigin: String,
}

#[derive(Debug, Deserialize, Clone, Copy)]
pub(crate) struct PlaneBounds {
    pub(crate) left: f64,
    pub(crate) bottom: f64,
    pub(crate) right: f64,
    pub(crate) top: f64,
}

#[derive(Debug, Deserialize, Clone, Copy)]
pub(crate) struct AtlasBounds {
    pub(crate) left: f64,
    pub(crate) bottom: f64,
    pub(crate) right: f64,
    pub(crate) top: f64,
}

#[derive(Debug, Deserialize)]
#[allow(unused, non_snake_case)]
struct Metrics {
    emSize: f64,
    lineHeight: f64,
    ascender: f64,
    descender: f64,
    underlineY: f64,
    underlineThickness: f64,
}

#[derive(Debug, Deserialize, Clone, Copy)]
#[allow(unused, non_snake_case)]
struct Glyph {
    unicode: u32,
    advance: f64,
    planeBounds: Option<PlaneBounds>,
    atlasBounds: Option<AtlasBounds>,
}

#[derive(Debug, Deserialize)]
#[allow(unused)]
pub(crate) struct FontData {
    pub(crate) atlas: Atlas,
    name: String,
    metrics: Metrics,
    glyphs: Vec<Glyph>,
}