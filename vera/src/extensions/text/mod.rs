use crate::{Colorization, Model, ToModel, Vertex};

mod font;

pub struct Text {
    string: String,
    // The text size
    size: f64,
    // The offset between characters. 0 for default.
    spacing: f64,
    spawn_time: f32,
    cursor: [f64; 2],
}

impl Text {
    /// Creates a new single-line text
    pub fn new(string: String, size: f64, spacing: f64, spawn_time: f32) -> Self {
        Self {
            string,
            size,
            spacing,
            spawn_time,
            cursor: [0.0, 0.0],
        }
    }
}

impl ToModel for Text {
    fn model(mut self) -> Model {
        let width = font::CMUNTI.atlas.width as f32;
        let height = font::CMUNTI.atlas.height as f32;
        let space_spacing= font::space_advance().unwrap();

        Model::from_vertices(
            self.string.chars().flat_map(|c| {
                if c==' ' {
                    dbg!("Space!");
                    self.cursor[0] += space_spacing + self.spacing;
                    vec![]
                } else {
                    let c2 = c as u32 as u8;
                    dbg!("Unicode for {}: {}", c, c2);
    
                    let char_bounds: (font::AtlasBounds, font::PlaneBounds, f64) = font::char_bounds(c2).unwrap();
    
    
    
                    let vec = vec![
                        Vertex::new().a(0.0).recolor(Colorization::ToColor(1.0, 1.0, 1.0, 1.0)).start_c((self.cursor[0] + char_bounds.1.left * self.size) as f32 / 10.0 + self.spawn_time).end_c((self.cursor[0] + char_bounds.1.left * self.size) as f32 / 10.0 + self.spawn_time + 0.5).pos((self.cursor[0] + char_bounds.1.left * self.size) as f32, (self.cursor[1] + char_bounds.1.top * self.size) as f32, 0.1-(self.cursor[0] / 100.0) as f32).tex(1, [char_bounds.0.left as f32 / width, 1.0 - (char_bounds.0.top as f32 / height)]),//[0.0, 1.0]
                        Vertex::new().a(0.0).recolor(Colorization::ToColor(1.0, 1.0, 1.0, 1.0)).start_c((self.cursor[0] + char_bounds.1.right * self.size) as f32 / 10.0 + self.spawn_time).end_c((self.cursor[0] + char_bounds.1.right * self.size) as f32 / 10.0 + self.spawn_time + 0.5).pos((self.cursor[0] + char_bounds.1.right * self.size) as f32, (self.cursor[1] + char_bounds.1.top * self.size) as f32, 0.1-(self.cursor[0] / 100.0) as f32).tex(1, [char_bounds.0.right as f32 / width, 1.0 - (char_bounds.0.top as f32 / height)]),//[1.0, 1.0]
                        Vertex::new().a(0.0).recolor(Colorization::ToColor(1.0, 1.0, 1.0, 1.0)).start_c((self.cursor[0] + char_bounds.1.left * self.size) as f32 / 10.0 + self.spawn_time).end_c((self.cursor[0] + char_bounds.1.left * self.size) as f32 / 10.0 + self.spawn_time + 0.5).pos((self.cursor[0] + char_bounds.1.left * self.size) as f32, (self.cursor[1] + char_bounds.1.bottom * self.size) as f32, 0.1-(self.cursor[0] / 100.0) as f32).tex(1, [char_bounds.0.left as f32 / width, 1.0 - (char_bounds.0.bottom as f32 / height)]),//[0.0, 0.0]
    
                        Vertex::new().a(0.0).recolor(Colorization::ToColor(1.0, 1.0, 1.0, 1.0)).start_c((self.cursor[0] + char_bounds.1.right * self.size) as f32 / 10.0 + self.spawn_time).end_c((self.cursor[0] + char_bounds.1.right * self.size) as f32 / 10.0 + self.spawn_time + 0.5).pos((self.cursor[0] + char_bounds.1.right * self.size) as f32, (self.cursor[1] + char_bounds.1.top * self.size) as f32, 0.1-(self.cursor[0] / 100.0) as f32).tex(1, [char_bounds.0.right as f32 / width, 1.0 - (char_bounds.0.top as f32 / height)]),//[1.0, 1.0]
                        Vertex::new().a(0.0).recolor(Colorization::ToColor(1.0, 1.0, 1.0, 1.0)).start_c((self.cursor[0] + char_bounds.1.left * self.size) as f32 / 10.0 + self.spawn_time).end_c((self.cursor[0] + char_bounds.1.left * self.size) as f32 / 10.0 + self.spawn_time + 0.5).pos((self.cursor[0] + char_bounds.1.left * self.size) as f32, (self.cursor[1] + char_bounds.1.bottom * self.size) as f32, 0.1-(self.cursor[0] / 100.0) as f32).tex(1, [char_bounds.0.left as f32 / width, 1.0 - (char_bounds.0.bottom as f32 / height)]),//[0.0, 0.0]
                        Vertex::new().a(0.0).recolor(Colorization::ToColor(1.0, 1.0, 1.0, 1.0)).start_c((self.cursor[0] + char_bounds.1.right * self.size) as f32 / 10.0 + self.spawn_time).end_c((self.cursor[0] + char_bounds.1.right * self.size) as f32 / 10.0 + self.spawn_time + 0.5).pos((self.cursor[0] + char_bounds.1.right * self.size) as f32, (self.cursor[1] + char_bounds.1.bottom * self.size) as f32, 0.1-(self.cursor[0] / 100.0) as f32).tex(1, [char_bounds.0.right as f32 / width, 1.0 - (char_bounds.0.bottom as f32 / height)]),//[1.0, 0.0]
                    ];
    
                    self.cursor[0] += (char_bounds.2 + self.spacing) * self.size;

                    vec
                }
            }).collect()
        )
    }
}