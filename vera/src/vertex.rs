//! Vertex creation and transformation.
//! This is an intermediate structure later reinterpreted by the Vera core.

use fastrand::f32;

use crate::{
    Evolution, Tf, Cl, Transformation, Colorization, 
    D_TRANSFORMATION_SPEED_EVOLUTION, D_TRANSFORMATION_START_TIME, D_TRANSFORMATION_END_TIME,
    D_COLORIZATION_SPEED_EVOLUTION, D_COLORIZATION_START_TIME, D_COLORIZATION_END_TIME,
};

/// A vertex with:
/// - its id. Changing it manually has no effect;
/// - the id of the entity it belongs to. Changing it manually has no effect;
/// - a 3D XYZ position;
/// - an RGBA color;
/// - (A Vec of transformations);
/// - (A Vec of colorizations);
///
/// Vertices enable building Triangles, which enable building all other shapes.
#[derive(Clone, Debug)]
pub struct Vertex {
    // Sent to GPU
    /// The position of the vertex, XYZW.
    position: [f32; 4],
    /// The color of the vertex, in RGBA format.
    color: [f32; 4],
    /// The coordinates of the vertex in the texture.
    tex_coord: [f32; 2],
    /// The id of the texture this vertex is linked to. For now:  0 if none, 1 if text, Ignored otherwise.
    tex_id: u32,

    // Treated in CPU
    t: Vec<Tf>,
    c: Vec<Cl>,
}
impl Vertex {
    /// A new default Vertex. Call this method to initialize a vertex, before transforming it.

    /// # Don't
    /// DO NOT call this function in multithreaded scenarios, as it calls static mut. See [the crate root](super).
    pub fn new() -> Self {
        unsafe {
            Self {
                position: [
                    super::D_VERTEX_POSITION[0],
                    super::D_VERTEX_POSITION[1],
                    super::D_VERTEX_POSITION[2],
                    1.0,
                ],
                color: if super::D_RANDOM_VERTEX_COLOR {
                    [f32(), f32(), f32(), super::D_VERTEX_ALPHA]
                } else {
                    [
                        super::D_VERTEX_COLOR[0],
                        super::D_VERTEX_COLOR[1],
                        super::D_VERTEX_COLOR[2],
                        super::D_VERTEX_ALPHA,
                    ]
                },
                tex_coord: [0.0, 0.0],
                tex_id: 0,
                t: vec![],
                c: vec![],
            }
        }
    }

    /// Creates a new vertex with the base position and color of `self`.
    pub fn duplicate(&self) -> Self{
        Self {
            position: self.position,
            color: self.color,
            tex_coord: [0.0, 0.0],
            tex_id: 0,
            t: vec![],
            c: vec![],
        }
    }

    /// Returns all the fields. Consumes `self`.
    pub fn own_fields(self) -> ([f32; 4], [f32; 4], [f32; 2], u32, Vec<Tf>, Vec<Cl>) {
        (
            self.position,
            self.color,
            self.tex_coord,
            self.tex_id,
            self.t,
            self.c,
        )
    }

    /// Reads the position data.
    pub fn read_position(&self) -> &[f32; 4] {
        &self.position
    }
    /// Reads the color data.
    pub fn read_color(&self) -> &[f32; 4] {
        &self.color
    }
    /// Reads the tex_coord data.
    pub fn read_tex_coord(&self) -> &[f32; 2] {
        &self.tex_coord
    }
    /// Reads the tex_id data.
    pub fn read_tex_id(&self) -> &u32 {
        &self.tex_id
    }
    /// Reads the c data.
    pub fn read_c(&self) -> &Vec<Cl> {
        &self.c
    }
    /// Reads the t data.
    pub fn read_tf(&self) -> &Vec<Tf> {
        &self.t
    }

    /// Gets a mutable reference to the position data.
    pub fn get_position(&mut self) -> &[f32; 4] {
        &self.position
    }
    /// Gets a mutable reference to the color data.
    pub fn get_color(&mut self) -> &mut [f32; 4] {
        &mut self.color
    }
    /// Gets a mutable reference to the tex_coord data.
    pub fn get_tex_coord(&mut self) -> &mut [f32; 2] {
        &mut self.tex_coord
    }
    /// Gets a mutable reference to the tex_id data.
    pub fn get_tex_id(&mut self) -> &mut u32 {
        &mut self.tex_id
    }
    /// Gets a mutable reference to the c data.
    pub fn get_c(&mut self) -> &mut Vec<Cl> {
        &mut self.c
    }
    /// Gets a mutable reference to the t data.
    pub fn get_tf(&mut self) -> &mut Vec<Tf> {
        &mut self.t
    }


    /// Modifies the position of the vertex to (x, y, z).
    pub fn pos(mut self, x: f32, y: f32, z: f32) -> Self {
        self.position = [x, y, z, 1.0];
        self
    }

    /// Modifies the red color channel of the vertex color.
    pub fn r(mut self, red: f32) -> Self {
        self.color[0] = red;
        self
    }

    /// Modifies the green color channel of the vertex color.
    pub fn g(mut self, green: f32) -> Self {
        self.color[1] = green;
        self
    }

    /// Modifies the blue color channel of the vertex color.
    pub fn b(mut self, blue: f32) -> Self {
        self.color[2] = blue;
        self
    }

    /// Modifies the alpha color channel of the vertex color.
    pub fn a(mut self, alpha: f32) -> Self {
        self.color[3] = alpha;
        self
    }

    /// Modifies the red, green and blue color channels of the vertex color.
    pub fn rgb(mut self, red: f32, green: f32, blue: f32) -> Self {
        self.color = [red, green, blue, self.color[3]];
        self
    }

    /// Modifies the red, green, blue and alpha color channels of the vertex color.
    pub fn rgba(mut self, red: f32, green: f32, blue: f32, alpha: f32) -> Self {
        self.color = [red, green, blue, alpha];
        self
    }

    // /// Sets the position of the vertex and ends the method calls pipe.
    // pub(crate) fn set_pos(&mut self, x: f32, y: f32, z: f32) {
    //     self.position = [x, y, z];
    // }

    /// Sets the color of the vertex and ends the method calls pipe.
    pub(crate) fn set_rgb(&mut self, red: f32, green: f32, blue: f32) {
        self.color = [red, green, blue, self.color[3]];
    }

    /// Sets the alpha of the vertex and ends the method calls pipe.
    pub(crate) fn set_alpha(&mut self, alpha: f32) {
        self.color[3] = alpha;
    }

    // -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    /// Sets the texture data for this vertex. Crate-visible for now as this is for text only
    pub(crate) fn tex(mut self, id: u32, coord: [f32; 2]) -> Self {
        self.tex_id = id;
        self.tex_coord = coord;
        self
    }

    // -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    /// Adds a new transformation to this vertex with default speed evolution, start time and end time.
    /// # Don't
    /// DO NOT call this function in multithreaded scenarios, as it calls static mut. See [the crate root](super).
    pub fn transform(mut self, transformation: Transformation) -> Self {
        self.t.push(Tf {
            t: transformation,
            e: unsafe { D_TRANSFORMATION_SPEED_EVOLUTION },
            start: unsafe { D_TRANSFORMATION_START_TIME },
            end: unsafe { D_TRANSFORMATION_END_TIME },
        });
        self
    }

    /// Adds a new color change to this vertex with default speed evolution, start time and end time.
    /// # Don't
    /// DO NOT call this function in multithreaded scenarios, as it calls static mut. See [the crate root](super).
    pub fn recolor(mut self, colorization: Colorization) -> Self {
        self.c.push(Cl {
            c: colorization,
            e: unsafe { D_COLORIZATION_SPEED_EVOLUTION },
            start: unsafe { D_COLORIZATION_START_TIME },
            end: unsafe { D_COLORIZATION_END_TIME },
        });
        self
    }

    /// Modifies the speed evolution of the latest colorization added.
    pub fn evolution_c(mut self, e: Evolution) -> Self {
        self.c.last_mut().unwrap().e = e;
        self
    }

    /// Modifies the start time of the latest colorization added.
    /// A start after an end will result in the colorization being instantaneous at start.
    pub fn start_c(mut self, start: f32) -> Self {
        self.c.last_mut().unwrap().start = start;
        self
    }

    /// Modifies the end time of the latest colorization added.
    /// An end before a start will result in the colorization being instantaneous at start.
    pub fn end_c(mut self, end: f32) -> Self {
        self.c.last_mut().unwrap().end = end;
        self
    }

    /// Modifies the speed evolution of the latest transformation added.
    pub fn evolution_t(mut self, e: Evolution) -> Self {
        self.t.last_mut().unwrap().e = e;
        self
    }

    /// Modifies the start time of the latest transformation added.
    /// A start after an end will result in the transformation being instantaneous at start.
    pub fn start_t(mut self, start: f32) -> Self {
        self.t.last_mut().unwrap().start = start;
        self
    }

    /// Modifies the end time of the latest transformation added.
    /// An end before a start will result in the transformation being instantaneous at start.
    pub fn end_t(mut self, end: f32) -> Self {
        self.t.last_mut().unwrap().end = end;
        self
    }
}
