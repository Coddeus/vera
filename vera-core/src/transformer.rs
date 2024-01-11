use std::f32::consts::PI;

use vera::{Transformation, Tf, Evolution};
use crate::Mat4;

impl Mat4 {
    /// Returns the transformation matrix for this transformation, with this advancement.
    fn from_t(transformation: Transformation, advancement: f32) -> Self {
        match transformation {
            Transformation::Scale(x, y, z) => Self::scale(x * advancement + (1.0 - advancement), y * advancement + (1.0 - advancement), z * advancement + (1.0 - advancement)),
            Transformation::Translate(x, y, z) => Self::translate(x * advancement, y * advancement, z * advancement),
            Transformation::RotateX(angle) => Self::rotate_x(angle * advancement),
            Transformation::RotateY(angle) => Self::rotate_y(angle * advancement),
            Transformation::RotateZ(angle) => Self::rotate_z(angle * advancement),

            Transformation::Lookat(eye_x, eye_y, eye_z, target_x, target_y, target_z, up_x, up_y, up_z) => Self::lookat(eye_x * advancement, eye_y * advancement, eye_z * advancement, target_x * advancement, target_y * advancement, target_z * advancement, up_x * advancement, up_y * advancement, up_z * advancement),

            // Transformation::Orthographic(l, r, b, t, n, f) => Self::project_orthographic(l * advancement, r * advancement, b * advancement, t * advancement, n * advancement, f * advancement),
            Transformation::Perspective(l, r, b, t, n, f) => Self::project_perspective(l * advancement, r * advancement, b * advancement, t * advancement, n * advancement, f * advancement),
            #[allow(unreachable_patterns)]
            _ => { println!("Transformation not implemented, ignoring."); Mat4::new() },
        }
    }
}

/// Intermediate type between Vertex/Model/View/Projection and buffer-sent Mat4.
/// Contains all transformations in the form of matrices.
pub(crate) struct Transformer {
    /// All matrices still needed every frame for `result` calculation.
    current: Vec<Tf>,

    /// The result of the previous matrices multiplication.
    /// Modified with current transformations before being sent to the buffer.
    result: Mat4,
}

impl Transformer {
    /// Creates a transformer from a vector of transformations.
    pub(crate) fn from_t(transformations: Vec<Tf>) -> Self {
        Self {
            current: transformations,

            result: Mat4::new(),
        }
    }

    /// Updates the transformer for view/projection transformations (interpolates every transformation into one), and returns the transformations matrix of the corresponding view/projection for `time`.
    pub(crate) fn update_vp(&mut self, time: f32) -> Mat4 {

        // let mut first = true;
        // let mut type_of_previous: Option<TransformType> = None;
        // 
        // self.current.retain(|&t| {
        //     if first {
        //         if t.end < time {
        //             self.result.mult(t.mat);
        //             self.previous.push(t);
        //             return false;
        //         } else {
        //             first = false;
        //             type_of_previous = Some(t.ty);
        //             return true;
        //         }
        //     }
        // 
        //     if type_of_previous == Some(t.ty) {
        //         if t.end < time {
        //             self.result.mult(t.mat);
        //             self.previous.push(t);
        //             return false;
        //         } else {
        //             return true;
        //         }
        //     }
        // 
        //     type_of_previous = None;
        // 
        //     true
        // });

        let mut buffer_matrix = self.result;
        self.current.iter().for_each(|tf| {
            let adv: f32 = advancement(tf.start, tf.end, time, tf.e);
            if adv>0.0 {
                buffer_matrix.interpolate(Mat4::from_t(tf.t, 1.0), adv);
            }
        });
        buffer_matrix
    }
}

/// Returns the *point of advancement* of `time` on the `start` to `end` journey, with the `e` evolution function.
/// The returned value is between 0.0 and 1.0, where 0.0 is the start and 1.0 is the end.
fn advancement(start: f32, end: f32, time: f32, e: Evolution) -> f32 {
    if start>=end {
        if time<end { return 0.0; }
        else { return 1.0; }
    }

    if time < start {
        return 0.0;
    }
    if time >= end {
        return 1.0
    }

    let init: f32 = (time-start)/(end-start);
    match e {
        Evolution::Linear => {
            init
        }
        Evolution::FastIn | Evolution::SlowOut => {
            (init * PI / 2.0).sin()
        }
        Evolution::FastOut | Evolution::SlowIn => {
            1.0 - (init * PI / 2.0).cos()
        }
        Evolution::FastMiddle | Evolution::SlowInOut => {
            (((init - 0.5) * PI).sin() + 1.0) / 2.0
        }
        Evolution::FastInOut | Evolution::SlowMiddle => {
            if init < 0.5 { (init * PI).sin() / 2.0 }
            else { 0.5 + (1.0 - (init * PI).sin()) / 2.0 }
        }
    }
}