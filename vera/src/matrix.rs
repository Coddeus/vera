/// A 4x4 matrix
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Mat4(pub [f32; 16]);

impl std::fmt::Debug for Mat4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[\n\t{}, {}, {}, {}, \n\t{}, {}, {}, {}, \n\t{}, {}, {}, {}, \n\t{}, {}, {}, {}, \n]", self.0[0], self.0[1], self.0[2], self.0[3], self.0[4], self.0[5], self.0[6], self.0[7], self.0[8], self.0[9], self.0[10], self.0[11], self.0[12], self.0[13], self.0[14], self.0[15])
    }
}

impl Mat4 {
    /// Build a new (identity) Mat4, which applies no transformations.
    pub fn new() -> Self {
        Mat4([
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ])
    }

    /// Returns a pointer to the matrix, readable by OpenGL.
    pub fn ptr(&self) -> *const f32 {
        self.0.as_ptr()
    }

    /// Multiplies this Mat4 (`self`) with another one (`mat`), further from the initial vertex position vector, so the resulting transformation will be the chaining of both matrices' transformations: first `self`, then `mat`.
    pub fn mult(&mut self, mat: Mat4) {
        *self = Mat4([
            mat.0[0]  * self.0[0] + mat.0[1]  * self.0[4] + mat.0[2]  * self.0[8] + mat.0[3]  * self.0[12]  , mat.0[0]  * self.0[1] + mat.0[1]  * self.0[5] + mat.0[2]  * self.0[9] + mat.0[3]  * self.0[13]  , mat.0[0] * self.0[2]  + mat.0[1] * self.0[6]  + mat.0[2] * self.0[10]  + mat.0[3] * self.0[14]   , mat.0[0] * self.0[3]  + mat.0[1] * self.0[7]  + mat.0[2] * self.0[11]  + mat.0[3] * self.0[15], 
            mat.0[4]  * self.0[0] + mat.0[5]  * self.0[4] + mat.0[6]  * self.0[8] + mat.0[7]  * self.0[12]  , mat.0[4]  * self.0[1] + mat.0[5]  * self.0[5] + mat.0[6]  * self.0[9] + mat.0[7]  * self.0[13]  , mat.0[4] * self.0[2]  + mat.0[5] * self.0[6]  + mat.0[6] * self.0[10]  + mat.0[7] * self.0[14]   , mat.0[4] * self.0[3]  + mat.0[5] * self.0[7]  + mat.0[6] * self.0[11]  + mat.0[7] * self.0[15], 
            mat.0[8]  * self.0[0] + mat.0[9]  * self.0[4] + mat.0[10] * self.0[8] + mat.0[11] * self.0[12]  , mat.0[8]  * self.0[1] + mat.0[9]  * self.0[5] + mat.0[10] * self.0[9] + mat.0[11] * self.0[13]  , mat.0[8] * self.0[2]  + mat.0[9] * self.0[6]  + mat.0[10] * self.0[10] + mat.0[11] * self.0[14]  , mat.0[8] * self.0[3]  + mat.0[9] * self.0[7]  + mat.0[10] * self.0[11] + mat.0[11] * self.0[15], 
            mat.0[12] * self.0[0] + mat.0[13] * self.0[4] + mat.0[14] * self.0[8] + mat.0[15] * self.0[12]  , mat.0[12] * self.0[1] + mat.0[13] * self.0[5] + mat.0[14] * self.0[9] + mat.0[15] * self.0[13]  , mat.0[12] * self.0[2] + mat.0[13] * self.0[6] + mat.0[14] * self.0[10] + mat.0[15] * self.0[14]  , mat.0[12] * self.0[3] + mat.0[13] * self.0[7] + mat.0[14] * self.0[11] + mat.0[15] * self.0[15], 
        ]);
    }

    /// Returns the linear interpolation between this matrix and the identity matrix, with the given *advancement* (between 0.0 and 1.0, where 0.0 is the identity matrix, and 1.0 is *self*).
    pub fn interpolate_idmat(&self, advancement: f32) -> Mat4 {
        Mat4([
            self.0[0] * advancement + 1.0 * (1.0-advancement)   ,   self.0[1] * advancement                             ,   self.0[2] * advancement                             ,   self.0[3] * advancement                             ,
            self.0[4] * advancement                             ,   self.0[5] * advancement + 1.0 * (1.0-advancement)   ,   self.0[6] * advancement                             ,   self.0[7] * advancement                             ,
            self.0[8] * advancement                             ,   self.0[9] * advancement                             ,   self.0[10] * advancement + 1.0 * (1.0-advancement)  ,   self.0[11] * advancement                            ,
            self.0[12] * advancement                            ,   self.0[13] * advancement                            ,   self.0[14] * advancement                            ,   self.0[15] * advancement + 1.0 * (1.0-advancement)  ,
        ])
    }

    /// Add a scale transformation to the Mat4, for each axis.
    /// The scale center is (0.0, 0.0, 0.0).
    pub fn scale(x_scale: f32, y_scale: f32, z_scale: f32) -> Self {
        Mat4([
            x_scale ,   0.0     ,   0.0     , 0.0 , 
            0.0     ,   y_scale ,   0.0     , 0.0 , 
            0.0     ,   0.0     ,   z_scale , 0.0 , 
            0.0     ,   0.0     ,   0.0     , 1.0 , 
        ])
    }

    /// Add a rotation transformation to the Mat4 around the X axis, clockiwse.
    /// The rotation center is (0.0, 0.0, 0.0).
    pub fn rotate_x(angle: f32) -> Self {
        Mat4([
            1.0     ,   0.0         ,   0.0         , 0.0 , 
            0.0     ,   angle.cos() ,   angle.sin() , 0.0 , 
            0.0     ,   -angle.sin(),   angle.cos() , 0.0 , 
            0.0     ,   0.0         ,   0.0         , 1.0 , 
        ])
    }

    /// Add a rotation transformation to the Mat4 around the Y axis, clockiwse.
    /// The rotation center is (0.0, 0.0, 0.0).
    pub fn rotate_y(angle: f32) -> Self {
        Mat4([
            angle.cos() ,   0.0 ,   angle.sin() , 0.0 , 
            0.0         ,   1.0 ,   0.0         , 0.0 , 
            -angle.sin(),   0.0 ,   angle.cos() , 0.0 , 
            0.0         ,   0.0 ,   0.0         , 1.0 , 
        ])
    }

    /// Add a rotation transformation to the Mat4 around the Z axis, clockiwse.
    /// The rotation center is (0.0, 0.0, 0.0).
    pub fn rotate_z(angle: f32) -> Self {
        Mat4([
            angle.cos() ,   angle.sin() ,   0.0 ,   0.0 , 
            -angle.sin(),   angle.cos() ,   0.0 ,   0.0 , 
            0.0         ,   0.0         ,   1.0 ,   0.0 , 
            0.0         ,   0.0         ,   0.0 ,   1.0 , 
        ])
    }

    /// Add a translation transformation to the Mat4.
    pub fn translate(x_move: f32, y_move: f32, z_move: f32) -> Self {
        Mat4([
            1.0 ,   0.0 ,   0.0 ,   x_move ,
            0.0 ,   1.0 ,   0.0 ,   y_move ,
            0.0 ,   0.0 ,   1.0 ,   z_move ,
            0.0 ,   0.0 ,   0.0 ,   1.0 ,
        ])
    }

    /// For view matrix. Moves the "camera" to (eye_x, eye_y, eye_z), looking at (target_x, target_y, target_z), with a "roll" roll angle, in radians.
    /// Replaces any earlier transformation to this Mat4.
    pub fn lookat(eye_x: f32, eye_y: f32, eye_z: f32, target_x: f32, target_y: f32, target_z: f32, mut up_x: f32, mut up_y: f32, mut up_z: f32,) -> Self {
        // Forward vector
        let (mut f_x, mut f_y, mut f_z) = (eye_x-target_x, eye_y-target_y, eye_z-target_z);
        let invlen = 1.0 / (f_x*f_x+f_y*f_y+f_z*f_z).sqrt();
        (f_x, f_y, f_z) = (f_x*invlen, f_y*invlen, f_z*invlen);
        
        // Left vector
        let (mut l_x, mut l_y, mut l_z) = (up_y*f_z - up_z*f_y, up_z*f_x - up_x*f_z, up_x*f_y - up_y*f_x);
        let invlen = 1.0 / (l_x*l_x+l_y*l_y+l_z*l_z).sqrt();
        (l_x, l_y, l_z) = (l_x*invlen, l_y*invlen, l_z*invlen);
        
        // Up vector correction
        (up_x, up_y, up_z) = (f_y*l_z - f_z*l_y, f_z*l_x - f_x*l_z, f_x*l_y - f_y*l_x);



        let mut mat = Self::translate(-eye_x, -eye_y, -eye_z);
        mat.mult(Mat4([
            l_x ,   l_y ,   l_z ,   0.0 ,
            up_x,   up_y,   up_z,   0.0 ,
            f_x ,   f_y ,   f_z ,   0.0 ,
            0.0 ,   0.0 ,   0.0 ,   1.0 ,
        ]));
        mat
    }

    /// For projection matrix. Defines an orthographic projection matrix with the given [left-right] - [top-bottom] - [near-far] frustrum.
    /// The default Frustrum is set to left-right: [-1.0, 1.0], top-bottom: [-1.0, 1.0], near-far: [-1.0, 1.0]
    /// Replaces any earlier transformation to this Mat4.
    pub fn project_orthographic(l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) -> Self {
        Mat4([
            2.0 / (r - l)   ,   0.0                 ,   0.0                 ,   -(r + l) / (r - l)  ,
            0.0             ,   2.0 / (t - b)       ,   0.0                 ,   -(t + b) / (t - b)  ,
            0.0             ,   0.0                 ,   -2.0 / (f - n)      ,   -(f + n) / (f - n)  ,
            0.0             ,   0.0                 ,   0.0                 ,    1.0                ,
        ])
    }

    /// For projection matrix. Defines an perspective projection matrix with the given [left-right] - [top-bottom] - [near-far] frustrum.
    /// The default Frustrum is set to left-right: [-1.0, 1.0], top-bottom: [-1.0, 1.0], near-far: [-1.0, 1.0]
    /// Replaces any earlier transformation to this Mat4.
    pub fn project_perspective(l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) -> Self {
        Mat4([
            2.0 * n/(r - l) ,   0.0                 ,   (r + l)/(r - l)     ,   0.0                     ,
            0.0             ,   2.0 * n / (t - b)   ,   (t + b) / (t - b)   ,   0.0                     ,
            0.0             ,   0.0                 ,   -(f + n) / (f - n)  ,   -(2.0 * f * n)/(f - n)  ,
            0.0             ,   0.0                 ,   -1.0                ,   0.0                     ,
        ])
    }
}