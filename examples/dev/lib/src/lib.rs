use std::f32::consts::PI;

use fastrand::f32;
use vera::{Input, MetaInput, View, Projection, Transformation, Model, Vertex, D_TRANSFORMATION_START_TIME, D_TRANSFORMATION_END_TIME, Colorization};

const PHI: f32 = 1.618033988749;
use itertools::Itertools;

// [Geodesic polyhedra](https://en.wikipedia.org/wiki/Geodesic_polyhedron) are a nice way of approximating a sphere with triangles.

#[no_mangle]
fn get() -> Input {
    unsafe {
        D_TRANSFORMATION_START_TIME = 0.;
        D_TRANSFORMATION_END_TIME = 0.;
    }

    let sphere = geodesic_sphere(icosahedron(), 4, 0., 1.); // geodesic_sphere(geodesic_sphere(geodesic_sphere(icosahedron(), 2, 0., 1.), 2, 1., 2.), 2, 2., 3.);

    Input {
        meta: MetaInput {
            bg: [0.3, 0.3, 0.3, 0.3],
            start: 0.0,
            end: 5.0,
        },
        m: vec![
            sphere
                .transform(Transformation::RotateX(PI/3.))
                .transform(Transformation::RotateY(PI/3.))
                .transform(Transformation::RotateY(4.*PI)).start_t(0.0).end_t(5.0),
        ],
        v: View::new().transform(Transformation::Lookat(0., 0., -6., 0., 0., 0., 0., -1., 0.)),
        p: Projection::new().transform(Transformation::Perspective(-0.1, 0.1, -0.1, 0.1, 0.2, 100.)),
    }
}


// Vizualize the geodesic vertices division indices like this (this is a frequency of 5): 
//      15  15
//     10 /\ 16
//    06 /\/\ 17
//   03 /\/\/\ 18
//  01 /\/\/\/\ 19
// 00 /\/\/\/\/\ 20

/// Modifies the given BASE_MODEL to a geodesic sphere with the given FREQUENCY.
/// Returns the modified model as well as the number of triangles in it in total.
/// - `base_model` is expected to contain only vertices, a number of them which is a multiple of 3, and to either be 1. tetrahedron, 2. octahedron, or 3. icosahedron. The center of this base model should be the world origin.
fn geodesic_sphere(base_model: Model, frequency: u32, start: f32, end: f32) -> Model {
    assert!(
        frequency != 0 && 
        base_model.models.is_empty() &&
        !base_model.vertices.is_empty() &&
        base_model.vertices.len()%3 == 0
    );

    // The number of lines the subtriangles form (in the case of layer 0, it is a point)
    let layers = frequency+1;
    let max_proj_i = match frequency%3 {
        0 => frequency*2/3,
        1 | 2 => frequency*2/3+1,
        _ => {0}
    };
    let sample_pos = base_model.vertices[0].position;
    let radius = (sample_pos[0].powi(2) + sample_pos[1].powi(2) + sample_pos[2].powi(2)).sqrt();
    let mut projection_multiplier: Vec<f32> = Vec::with_capacity(max_proj_i as usize); // Not used by the 3 corner vertices, so indexing will need `-1`.

    let all_vertices: Vec<Vertex> = base_model.vertices.into_iter().chunks(3).into_iter().enumerate().flat_map(|(base_triangle_i, mut v)| {
        // Finding all vertices
        // The following iterates through each layer and each of its points to find the vertices and their coordinates.
        let face_red = (f32()+1.)/2.;

        let mut vertices: Vec<Vertex> = Vec::with_capacity((frequency * (frequency+1) / 2) as usize);

        let v0 = v.next().unwrap();
        let v1 = v.next().unwrap();
        let v2 = v.next().unwrap();

        // Layer 0
        vertices.push(v0.clone().rgba(face_red, 0., 0.2, 0.9).recolor(Colorization::ToColor(0., 0., 0., 0.9)).start_c(start).end_c(end));

        // Layers 1 to `frequency`-1
        // => Done only if (frequency > 1 (<=> max_proj_i > 0)).
        if base_triangle_i==0 { // First time, calculates the projection multipliers
            for layer_i in 1..layers-1 {
                let layer_mult_base_i = layer_i.min(layers-1-layer_i);
                let layer_points = layer_i+1;

                let unprojected_first: Vertex =  new_vertex_unprojected(&v0, &v1, layer_i, layers-1);
                let unprojected_last: Vertex =  new_vertex_unprojected(&v0, &v2, layer_i, layers-1);
                let first: Vertex;
                let last: Vertex;

                if let Some(&mult) = projection_multiplier.get((layer_mult_base_i-1) as usize) {
                    first = unprojected_first.clone().rgba(face_red, 0., 0., 0.9).transform(Transformation::Scale(mult, mult, mult)).start_t(start).end_t(end).recolor(Colorization::ToColor(layer_mult_base_i as f32/max_proj_i as f32, 0., 0., 0.9)).start_c(start).end_c(end);
                    last = unprojected_last.clone().rgba(face_red, 0., 0., 0.9).transform(Transformation::Scale(mult, mult, mult)).start_t(start).end_t(end).recolor(Colorization::ToColor(layer_mult_base_i as f32/max_proj_i as f32, 0., 0., 0.9)).start_c(start).end_c(end);
                } else {
                    let new_mult = radius/(unprojected_first.position[0].powi(2) + unprojected_first.position[1].powi(2) + unprojected_first.position[2].powi(2)).sqrt();
                    projection_multiplier.push(new_mult);

                    first = unprojected_first.clone().rgba(face_red, 0., 0., 0.9).transform(Transformation::Scale(new_mult, new_mult, new_mult)).start_t(start).end_t(end).recolor(Colorization::ToColor(layer_mult_base_i as f32/max_proj_i as f32, 0., 0., 0.9)).start_c(start).end_c(end);
                    last = unprojected_last.clone().rgba(face_red, 0., 0., 0.9).transform(Transformation::Scale(new_mult, new_mult, new_mult)).start_t(start).end_t(end).recolor(Colorization::ToColor(layer_mult_base_i as f32/max_proj_i as f32, 0., 0., 0.9)).start_c(start).end_c(end);
                }

                // Point 0
                vertices.push(first);

                // Points 1 to penultimate
                for point_i in 1..layer_points-1 { 
                    let point_mult_i = layer_mult_base_i + point_i.min(layer_points-1-point_i);

                    if let Some(&mult) = projection_multiplier.get((point_mult_i-1) as usize) {
                        vertices.push(new_vertex_unprojected(&unprojected_first, &unprojected_last, point_i, layer_points-1).rgba(face_red, 0., 0., 0.9).transform(Transformation::Scale(mult, mult, mult)).start_t(start).end_t(end).recolor(Colorization::ToColor(layer_mult_base_i as f32/max_proj_i as f32, 0., 0., 0.9)).start_c(start).end_c(end))
                    } else {
                        let unprojected_point = new_vertex_unprojected(&unprojected_first, &unprojected_last, point_i, layer_points-1);

                        let new_mult = radius/(unprojected_point.position[0].powi(2) + unprojected_point.position[1].powi(2) + unprojected_point.position[2].powi(2)).sqrt();
                        projection_multiplier.push(new_mult);

                        vertices.push(vertex_multiplied(&unprojected_point, new_mult).rgba(face_red, 0., 0., 0.9).transform(Transformation::Scale(new_mult, new_mult, new_mult)).start_t(start).end_t(end).recolor(Colorization::ToColor(point_mult_i as f32/max_proj_i as f32, 0., 0., 0.9)).start_c(start).end_c(end));
                    }

                }

                // Last point
                vertices.push(last);
            }
        } else { // Every other time, optimized.
            for layer_i in 1..layers-1 {
                let layer_mult_base_i = layer_i.min(layers-1-layer_i);
                let layer_points = layer_i+1;

                let unprojected_first: Vertex = new_vertex_unprojected(&v0, &v1, layer_i, layers-1);
                let unprojected_last: Vertex = new_vertex_unprojected(&v0, &v2, layer_i, layers-1);
                let first: Vertex;
                let last: Vertex;

                if let Some(&mult) = projection_multiplier.get((layer_mult_base_i-1) as usize) {
                    first = unprojected_first.clone().rgba(face_red, 0., 0., 0.9).transform(Transformation::Scale(mult, mult, mult)).start_t(start).end_t(end).recolor(Colorization::ToColor(layer_mult_base_i as f32/max_proj_i as f32, 0., 0., 0.9)).start_c(start).end_c(end);
                    last = unprojected_last.clone().rgba(face_red, 0., 0., 0.9).transform(Transformation::Scale(mult, mult, mult)).start_t(start).end_t(end).recolor(Colorization::ToColor(layer_mult_base_i as f32/max_proj_i as f32, 0., 0., 0.9)).start_c(start).end_c(end);
                } else {
                    panic!("Projection multiplier doesn't exist! (first or last of layer)");
                }

                // Point 0
                vertices.push(first);

                // Points 1 to penultimate
                for point_i in 1..layer_points-1 {
                    let point_mult_i = layer_mult_base_i + point_i.min(layer_points-1-point_i);

                    if let Some(&mult) = projection_multiplier.get((point_mult_i-1) as usize) {
                        vertices.push(new_vertex_unprojected(&unprojected_first, &unprojected_last, point_i, layer_points-1).rgba(face_red, 0., 0., 0.9).transform(Transformation::Scale(mult, mult, mult)).start_t(start).end_t(end).recolor(Colorization::ToColor(point_mult_i as f32/max_proj_i as f32, 0., 0., 0.9)).start_c(start).end_c(end))
                    } else {
                        panic!("Projection multiplier doesn't exist! (inner point in layer)");
                    }

                }

                // Last point
                vertices.push(last);
            }
        }

        // Last layer

        // Point 0
        vertices.push(v1.clone().rgba(face_red, 0., 0., 0.9).recolor(Colorization::ToColor(0., 0., 0., 0.9)).start_c(start).end_c(end));
        
        // Points 1 to penultimate
        let layer_points = layers;
        for point_i in 1..layer_points-1 {
            let point_mult_i = point_i.min(layer_points-1-point_i);

            if let Some(&mult) = projection_multiplier.get((point_mult_i-1) as usize) {
                vertices.push(new_vertex_unprojected(&v1, &v2, point_i, layer_points-1).rgba(face_red, 0., 0., 0.9).transform(Transformation::Scale(mult, mult, mult)).start_t(start).end_t(end).recolor(Colorization::ToColor(point_mult_i as f32/max_proj_i as f32, 0., 0., 0.9)).start_c(start).end_c(end))
            } else {
                panic!("Projection multiplier doesn't exist! (inner point in layer)");
            }

        }

        // Last point
        vertices.push(v2.rgba(face_red, 0., 0., 0.9).recolor(Colorization::ToColor(0., 0., 0., 0.9)).start_c(start).end_c(end));


        // -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

        // Organizing all vertices into triangles

        let mut triangles: Vec<Vertex> = vec![];

        for layer_i in 0..layers-1 {
            let first_point = (layer_i) * (layer_i+1) / 2;
            let next_first_point = (layer_i+1) * (layer_i+2) / 2;
            // First triangle of the layer
            triangles.push(vertices[(first_point) as usize].clone());
            triangles.push(vertices[(next_first_point) as usize].clone());
            triangles.push(vertices[(next_first_point + 1) as usize].clone());

            // Every other pair of triangles in the layer (the 2 triangles that vertex belongs to, that are on the right of that vertex on the above pyramid vizualization)
            for point_added_i in 0..(next_first_point-first_point-1)  {
                triangles.push(vertices[(point_added_i + first_point) as usize].clone());
                triangles.push(vertices[(point_added_i + first_point + 1) as usize].clone());
                triangles.push(vertices[(point_added_i + next_first_point + 1) as usize].clone());

                triangles.push(vertices[(point_added_i + first_point + 1) as usize].clone());
                triangles.push(vertices[(point_added_i + next_first_point + 1) as usize].clone());
                triangles.push(vertices[(point_added_i + next_first_point + 2) as usize].clone());
            }
        }
        
        triangles
    }).collect();

    let faces = all_vertices.len() as u32 / 3;
    
    Model::from_vertices(all_vertices)
}

// /// Returns a vertex on the line which goes from V1 to V2, at V/W, projected on the geodesic sphere using the given MULTiplier (= radius/distance_to_origin, considering the model is centered).
// fn new_vertex(v1: &Vertex, v2: &Vertex, v: u32, w: u32, mult: f32) -> Vertex {
//     assert!(v<w);
//     let x = (v1.position[0] * (w-v) as f32 + v2.position[0] * v as f32) / w as f32;
//     let y = (v1.position[1] * (w-v) as f32 + v2.position[1] * v as f32) / w as f32;
//     let z = (v1.position[2] * (w-v) as f32 + v2.position[2] * v as f32) / w as f32;
// 
//     Vertex::new().pos(x * mult, y * mult, z * mult)
// }
/// Returns a vertex on the line which goes from V1 to V2, at V/W, to be later projected on the geodesic sphere with the given RADius, by multiplying its xyz coordinates by the returned rojection MULTiplier that was used.
fn new_vertex_unprojected(v1: &Vertex, v2: &Vertex, v: u32, w: u32) -> Vertex {
    assert!(v<w);
    let x = (v1.position[0] * (w-v) as f32 + v2.position[0] * v as f32) / w as f32;
    let y = (v1.position[1] * (w-v) as f32 + v2.position[1] * v as f32) / w as f32;
    let z = (v1.position[2] * (w-v) as f32 + v2.position[2] * v as f32) / w as f32;

    Vertex::new().pos(x, y, z)
}
/// Returns a projected vertex from an unprojected vertex, given the multiplier.
fn vertex_multiplied(unprojected_v: &Vertex, mult: f32) -> Vertex {
    Vertex::new().pos(unprojected_v.position[0] * mult, unprojected_v.position[1] * mult, unprojected_v.position[2] * mult)
}

/// The base Icosahedron
fn icosahedron() -> Model {
    Model::from_vertices(vec![
        Vertex::new().pos(0., 1., PHI).rgba(f32(), 0., 0.2, 0.9)      ,Vertex::new().pos(1., PHI, 0.).rgba(f32(), 0., 0.2, 0.9)     ,Vertex::new().pos(-1., PHI, 0.).rgba(f32(), 0., 0.2, 0.9)     ,
        Vertex::new().pos(0., 1., PHI).rgba(f32(), 0., 0.2, 0.9)      ,Vertex::new().pos(-1., PHI, 0.).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(-PHI, 0., 1.).rgba(f32(), 0., 0.2, 0.9)     ,
        Vertex::new().pos(0., 1., PHI).rgba(f32(), 0., 0.2, 0.9)      ,Vertex::new().pos(-PHI, 0., 1.).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(0., -1., PHI).rgba(f32(), 0., 0.2, 0.9)     ,
        Vertex::new().pos(0., 1., PHI).rgba(f32(), 0., 0.2, 0.9)      ,Vertex::new().pos(0., -1., PHI).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(PHI, 0., 1.).rgba(f32(), 0., 0.2, 0.9)      ,
        Vertex::new().pos(0., 1., PHI).rgba(f32(), 0., 0.2, 0.9)      ,Vertex::new().pos(PHI, 0., 1.).rgba(f32(), 0., 0.2, 0.9)     ,Vertex::new().pos(1., PHI, 0.).rgba(f32(), 0., 0.2, 0.9)      ,

        Vertex::new().pos(0., -1., PHI).rgba(f32(), 0., 0.2, 0.9)     ,Vertex::new().pos(-PHI, 0., 1.).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(-1., -PHI, 0.).rgba(f32(), 0., 0.2, 0.9)    ,
        Vertex::new().pos(0., -1., PHI).rgba(f32(), 0., 0.2, 0.9)     ,Vertex::new().pos(-1., -PHI, 0.).rgba(f32(), 0., 0.2, 0.9)   ,Vertex::new().pos(1., -PHI, 0.).rgba(f32(), 0., 0.2, 0.9)     ,
        Vertex::new().pos(0., -1., PHI).rgba(f32(), 0., 0.2, 0.9)     ,Vertex::new().pos(1., -PHI, 0.).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(PHI, 0., 1.).rgba(f32(), 0., 0.2, 0.9)      ,
        Vertex::new().pos(0., -1., -PHI).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(0., 1., -PHI).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(PHI, 0., -1.).rgba(f32(), 0., 0.2, 0.9)     ,
        Vertex::new().pos(0., -1., -PHI).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(PHI, 0., -1.).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(1., -PHI, 0.).rgba(f32(), 0., 0.2, 0.9)     ,

        Vertex::new().pos(0., -1., -PHI).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(1., -PHI, 0.).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(-1., -PHI, 0.).rgba(f32(), 0., 0.2, 0.9)    ,
        Vertex::new().pos(0., -1., -PHI).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(-1., -PHI, 0.).rgba(f32(), 0., 0.2, 0.9)   ,Vertex::new().pos(-PHI, 0., -1.).rgba(f32(), 0., 0.2, 0.9)    ,
        Vertex::new().pos(0., -1., -PHI).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(-PHI, 0., -1.).rgba(f32(), 0., 0.2, 0.9)   ,Vertex::new().pos(0., 1., -PHI).rgba(f32(), 0., 0.2, 0.9)     ,
        Vertex::new().pos(0., 1., -PHI).rgba(f32(), 0., 0.2, 0.9)     ,Vertex::new().pos(-PHI, 0., -1.).rgba(f32(), 0., 0.2, 0.9)   ,Vertex::new().pos(-1., PHI, 0.).rgba(f32(), 0., 0.2, 0.9)     ,
        Vertex::new().pos(0., 1., -PHI).rgba(f32(), 0., 0.2, 0.9)     ,Vertex::new().pos(-1., PHI, 0.).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(1., PHI, 0.).rgba(f32(), 0., 0.2, 0.9)      ,

        Vertex::new().pos(0., 1., -PHI).rgba(f32(), 0., 0.2, 0.9)     ,Vertex::new().pos(1., PHI, 0.).rgba(f32(), 0., 0.2, 0.9)     ,Vertex::new().pos(PHI, 0., -1.).rgba(f32(), 0., 0.2, 0.9)     ,
        Vertex::new().pos(PHI, 0., 1.).rgba(f32(), 0., 0.2, 0.9)      ,Vertex::new().pos(1., -PHI, 0.).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(PHI, 0., -1.).rgba(f32(), 0., 0.2, 0.9)     ,
        Vertex::new().pos(PHI, 0., 1.).rgba(f32(), 0., 0.2, 0.9)      ,Vertex::new().pos(PHI, 0., -1.).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(1., PHI, 0.).rgba(f32(), 0., 0.2, 0.9)      ,
        Vertex::new().pos(-PHI, 0., 1.).rgba(f32(), 0., 0.2, 0.9)     ,Vertex::new().pos(-1., PHI, 0.).rgba(f32(), 0., 0.2, 0.9)    ,Vertex::new().pos(-PHI, 0., -1.).rgba(f32(), 0., 0.2, 0.9)    ,
        Vertex::new().pos(-PHI, 0., 1.).rgba(f32(), 0., 0.2, 0.9)     ,Vertex::new().pos(-PHI, 0., -1.).rgba(f32(), 0., 0.2, 0.9)   ,Vertex::new().pos(-1., -PHI, 0.).rgba(f32(), 0., 0.2, 0.9)    ,
    ])
}