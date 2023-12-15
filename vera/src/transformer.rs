use vera_shapes::{Transformation, Tf, Evolution};
use crate::Mat4;

/// Lighter enum for transformation types.
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum TransformType {
    Scale,
    Translate,
    RotateX,
    RotateY,
    RotateZ,

    Lookat,

    Orthographic,
    Perspective,

    Unimplemented,
}

impl From<&Transformation> for TransformType {
    fn from(value: &Transformation) -> Self {
        match value {
            Transformation::Scale(_, _, _) => TransformType::Scale,
            Transformation::Translate(_, _, _) => TransformType::Translate,
            Transformation::RotateX(_) => TransformType::RotateX,
            Transformation::RotateY(_) => TransformType::RotateY,
            Transformation::RotateZ(_) => TransformType::RotateZ,

            Transformation::Lookat(_, _, _, _, _, _, _, _, _) => TransformType::Lookat,

            Transformation::Orthographic(_, _, _, _, _, _) => TransformType::Orthographic,
            Transformation::Perspective(_, _, _, _, _, _) => TransformType::Perspective,

            _ => TransformType::Unimplemented,
        }
    }
}

impl From<Transformation> for Mat4 {
    fn from(value: Transformation) -> Self {
        match value {
            Transformation::Scale(x, y, z) => Mat4::scale(x, y, z),
            Transformation::Translate(x, y, z) => Mat4::translate(x, y, z),
            Transformation::RotateX(angle) => Mat4::rotate_x(angle),
            Transformation::RotateY(angle) => Mat4::rotate_y(angle),
            Transformation::RotateZ(angle) => Mat4::rotate_z(angle),

            Transformation::Lookat(eye_x, eye_y, eye_z, target_x, target_y, target_z, up_x, up_y, up_z) => Mat4::lookat(eye_x, eye_y, eye_z, target_x, target_y, target_z, up_x, up_y, up_z),

            Transformation::Orthographic(l, r, b, t, n, f) => Mat4::project_orthographic(l, r, b, t, n, f),
            Transformation::Perspective(l, r, b, t, n, f) => Mat4::project_perspective(l, r, b, t, n, f),
            _ => Mat4::new(),
        }
    }
}

/// A transformation
#[derive(Clone, Copy)]
struct TransformMatrix {
    /// The type of transformation. Stored for update optimization.
    pub ty: TransformType,
    /// The transformation matrix.
    pub mat: Mat4,
    /// The speed evolution of the transformation.
    pub e: Evolution,
    /// The start time of the transformation.
    pub start: f32,
    /// The duration of the transformation.
    pub end: f32,
}

impl From<Tf> for TransformMatrix {
    fn from(value: Tf) -> Self {
        TransformMatrix {
            ty: (&value.t).into(),
            mat: value.t.into(),
            e: value.e,
            start: value.start,
            end: value.end,
        }
    }
}

/// Intermediate type between Vertex/Model/View/Projection and buffer-sent Mat4.
/// Contains all transformations in the form of matrices.
pub(crate) struct Transformer {
    /// Matrices already fully applied to `result`, not used anymore. // TODO Remove?
    previous: Vec<TransformMatrix>,
    /// All matrices still needed every frame for `result` calculation.
    current: Vec<TransformMatrix>,

    /// The result of the previous matrices multiplication.
    /// Modified with current transformations before being sent to the buffer.
    result: Mat4,
}

impl Transformer {
    pub(crate) fn empty() -> Self {
        Self {
            previous: vec![],
            current: vec![],

            result: Mat4::new(),
        }
    }

    /// Creates a transformer from a vector of transformations.
    pub(crate) fn from_t(transformations: Vec<Tf>) -> Self {
        Self {
            previous: Vec::with_capacity(transformations.len()),
            current: transformations
                .into_iter()
                .map(|tf| { tf.into() })
                .collect::<Vec<TransformMatrix>>(),

            result: Mat4::new(),
        }
    }

    // TODO update with drain_filter() when stable
    // The storing order cannot change, except if consecutive transformations have the same type.
    // Faster reset in a loop-like action ?

    /// Updates the transformer, and returns the transformations matrix of the corresponding vertex for `time`.
    pub(crate) fn update(&mut self, time: f32) -> Mat4 {
        let mut first = true;
        let mut type_of_previous: Option<TransformType> = None;

        self.current.retain(|&t| {
            if first {
                if t.end < time {
                    self.result.mult(t.mat);
                    self.previous.push(t);
                    return false;
                } else {
                    first = false;
                    type_of_previous == Some(t.ty);
                    return true;
                }
            }

            if type_of_previous == Some(t.ty) {
                if t.end < time {
                    self.result.mult(t.mat);
                    self.previous.push(t);
                    return false;
                } else {
                    return true;
                }
            }

            type_of_previous = None;

            true
        });

        let mut buffer_matrix = self.result;
        self.current.iter().for_each(|tf| { buffer_matrix.mult(tf.mat.mult_float(evolution(tf.start, tf.end, time, tf.e))); });
        buffer_matrix
    }
}

/// Returns the *point of advancement* of `time` on the `start` to `end` journey, with the `e` evolution function.
/// The returned value is between 0.0 and 1.0, where 0.0 is the start and 1.0 is the end.
fn evolution(start: f32, end: f32, time: f32, e: Evolution) -> f32 {
    if start>=end {
        if time<start { return 0.0; }
        else { return 1.0; }
    }

    match e {
        Evolution::Linear => {
            (time-start)/(end-start)
        }
        Evolution::Unimplemented => {
            println!("Unknown evolution, defaulting to Linear");
            (time-start)/(end-start)
        }
    }
}