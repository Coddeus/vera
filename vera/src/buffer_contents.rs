use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};
use winit::dpi::PhysicalSize;
use vera_shapes;

/// A base vertex for Vera, meant to be given as input to the graphics pipeline.
#[derive(BufferContents, Vertex, Debug)]
#[repr(C)]
pub struct Veratex {
    /// The (x, y) [normalized-square-centered](broken link) coordinates of the vertex.
    #[format(R32G32_SFLOAT)]
    pub(crate) position: [f32; 2],
    /// The id of the entity. Used and overriden when calling `vera::Vera::set()`
    #[format(R32_UINT)]
    pub(crate) entity_id: u32,
}
impl vera_shapes::Vertex for Veratex {
    /// A new `Veratex` with (x, y) 2D coordinates.
    fn new(x: f32, y: f32) -> Self {
        Veratex { position: [x, y], ..Default::default() }
    }
}
impl Default for Veratex {
    fn default() -> Veratex {
        Veratex {
            position: [-0.5, 0.5],
            entity_id: 0,
        }
    }
}

/// General-purpose uniform data
#[derive(Debug, Clone, BufferContents)]
#[repr(C)]
pub(crate) struct GeneralData {
    /// The transformation matrix of the viewport (enables translation, rotation, scaling)
    /// ```
    /// RotateCos * ScaleX  , -RotateSin        , TranslateX, alignment (unused),
    /// RotateSin           , RotateCos * ScaleY, TranslateY, alignment (unused),
    /// 0.0                 , 0.0               , 1.0       , alignment (unused),
    /// ```
    pub(crate) view_matrix: [f32 ; 12],
    // projection_matrix: [f32 ; 12],
    /// The viewport resolution. Used to unstretch the shapes within the viewport.
    /// ```
    pub(crate) resolution: [f32 ; 2],
    /// The time elapsed since the beginning.
    pub(crate) time: f32
}
impl GeneralData {
    /// Returns data applying no transformation.
    pub(crate) fn empty() -> Self {
        GeneralData { 
            view_matrix: [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ],
            resolution: [1.0, 1.0],
            time: 0.0
        }
    }
    /// Returns data applying no transformation.
    pub(crate) fn from_resolution(inner_size: PhysicalSize<u32>) -> Self {
        GeneralData { 
            view_matrix: [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ],
            resolution: [inner_size.width as f32, inner_size.height as f32],
            time: 0.0
        }
    }
    /// Sets the resolution to the given window size
    pub(crate) fn resolution(&mut self, inner_size: PhysicalSize<u32>) {
        self.resolution = [inner_size.width as f32, inner_size.height as f32];
    }
}

/// Uniform data for one entity
#[derive(Debug, Clone, BufferContents)]
#[repr(C)]
pub(crate) struct EntitiesData {
    /// The transformation matrix of an element (enables translation, rotation, scaling)
    /// ```
    /// RotateCos * ScaleX  , -RotateSin        , TranslateX, alignment(unused),
    /// RotateSin           , RotateCos * ScaleY, TranslateY, alignment(unused),
    /// 0.0                 , 0.0               , 1.0       , alignment(unused),
    /// ```
    model_matrix: [f32 ; 12],

    // color: [f32 ; 4],
}
impl EntitiesData {
    /// Returns an identity matrix, applying no transformation.
    pub(crate) fn empty() -> Self {
        EntitiesData { 
            model_matrix: [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ],
        }
    }
}