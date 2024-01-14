use std::f32::consts::PI;

use fastrand::f32;
use vera::{Input, MetaInput, View, Projection, Transformation, Model, Vertex, D_TRANSFORMATION_START_TIME, D_TRANSFORMATION_END_TIME, Colorization/*, Evolution*/};

const PHI: f32 = 1.618033988749;
use itertools::Itertools;
use vera_core::Vera;


// [Geodesic polyhedra](https://en.wikipedia.org/wiki/Geodesic_polyhedron) are a nice way of approximating a sphere with triangles.
// Overall followed the method of [Dirk Bertels's paper](https://www.dirkbertels.net/computing/pentaDome.php).

fn main() {
    let mut v = Vera::init(get());
    v.show();
}
fn get() -> Input {
    unsafe {
        D_TRANSFORMATION_START_TIME = 0.;
        D_TRANSFORMATION_END_TIME = 0.;
    }

    // Frequencies can generally go over 100, it's just a bit of loading after that.
    let sphere1 = geodesic_sphere(tetrahedron(), 20, 2., 5.); 
    let sphere2 = geodesic_sphere(octahedron(), 20, 6., 9.);
    let sphere3 = geodesic_sphere(icosahedron(), 20, 10., 13.);

    Input {
        meta: MetaInput {
            bg: [0.3, 0.3, 0.3, 0.3],
            start: 0.0,
            end: 14.0,
        },
        m: vec![
            sphere1
                .transform(Transformation::Scale(0.5, 0.5, 0.5)).start_t(0.0).end_t(0.0)
                .transform(Transformation::Scale(2., 2., 2.)).start_t(1.0).end_t(2.0)
                .transform(Transformation::Scale(0.1, 0.1, 0.1)).start_t(5.0).end_t(6.0)
                .transform(Transformation::RotateY(2.*PI)).start_t(2.0).end_t(5.0),
            sphere2
                .transform(Transformation::Scale(0.1, 0.1, 0.1)).start_t(0.0).end_t(0.0)
                .transform(Transformation::Scale(10., 10., 10.)).start_t(5.0).end_t(6.0)
                .transform(Transformation::Scale(0.1, 0.1, 0.1)).start_t(9.0).end_t(10.0)
                .transform(Transformation::RotateY(2.*PI)).start_t(6.0).end_t(9.0),
            sphere3
                .transform(Transformation::Scale(0.1, 0.1, 0.1)).start_t(0.0).end_t(0.0)
                .transform(Transformation::Scale(10., 10., 10.)).start_t(9.0).end_t(10.0)
                .transform(Transformation::Scale(0.2, 0.2, 0.2)).start_t(13.0).end_t(14.0)
                .transform(Transformation::RotateY(2.*PI)).start_t(10.0).end_t(13.0),
        ],
        v: View::new().transform(Transformation::Lookat(0., 0., -3., 0., 0., 0., 0., -1., 0.)),
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

    let sample_pos = base_model.vertices[0].position;
    let radius = (sample_pos[0].powi(2) + sample_pos[1].powi(2) + sample_pos[2].powi(2)).sqrt();

    let subpoints = ((frequency+1) * (frequency+2) / 2) as usize;
    let mut projection_multipliers: Vec<f32> = Vec::with_capacity(subpoints);

    let all_vertices: Vec<Vertex> = base_model.vertices.into_iter().chunks(3).into_iter().enumerate().flat_map(|(base_triangle_i, mut v)| {
        // Finding all vertices
        // The following iterates through each layer and each of its points to find the vertices and their coordinates.
        let face_red = (f32()+1.)/2.;

        let mut vertices: Vec<Vertex> = Vec::with_capacity(subpoints);
        let mut subvertex_i = 0;

        let v0 = v.next().unwrap();
        let v1 = v.next().unwrap();
        let v2 = v.next().unwrap();
        if base_triangle_i==0 { // First time, calculates the projection multipliers

            // Layer 0
            vertices.push(v0.clone());
            projection_multipliers.push(1.0);
    
            // Layers 1 to `frequency`-1
            // => Done only if (frequency > 1 (<=> max_proj_i > 0)).
            for layer_i in 1..layers-1 {
                let layer_points = layer_i+1;
                let unprojected_first: Vertex = new_vertex_unprojected(&v0, &v1, layer_i, layers-1);
                let unprojected_last: Vertex = new_vertex_unprojected(&v0, &v2, layer_i, layers-1);
                let new_mult = radius/(unprojected_first.position[0].powi(2) + unprojected_first.position[1].powi(2) + unprojected_first.position[2].powi(2)).sqrt();

                // Point 0
                vertices.push(unprojected_first.duplicate().transform(Transformation::Scale(new_mult, new_mult, new_mult)).start_t(start).end_t(end));
                projection_multipliers.push(new_mult);

                // Points 1 to penultimate
                for point_i in 1..layer_points-1 {
                    let unprojected_point = new_vertex_unprojected(&unprojected_first, &unprojected_last, point_i, layer_points-1);
                    let new_mult = radius/(unprojected_point.position[0].powi(2) + unprojected_point.position[1].powi(2) + unprojected_point.position[2].powi(2)).sqrt();
                    vertices.push(unprojected_point.transform(Transformation::Scale(new_mult, new_mult, new_mult)).start_t(start).end_t(end));
                    projection_multipliers.push(new_mult);
                }

                // Last point
                vertices.push(unprojected_last.duplicate().transform(Transformation::Scale(new_mult, new_mult, new_mult)).start_t(start).end_t(end));
                projection_multipliers.push(new_mult);
            }

            // Last layer
    
            // Point 0
            vertices.push(v1.duplicate());
            projection_multipliers.push(1.0);
            
            // Points 1 to penultimate
            let layer_points = layers;
            for point_i in 1..layer_points-1 {
                let unprojected_point = new_vertex_unprojected(&v1, &v2, point_i, layer_points-1);
                let new_mult = radius/(unprojected_point.position[0].powi(2) + unprojected_point.position[1].powi(2) + unprojected_point.position[2].powi(2)).sqrt();
                vertices.push(unprojected_point.transform(Transformation::Scale(new_mult, new_mult, new_mult)).start_t(start).end_t(end));
                projection_multipliers.push(new_mult);
            }
    
            // Last point
            vertices.push(v2);
            projection_multipliers.push(1.0);

        } else { // Every other time, optimized.
            // Layer 0
            vertices.push(v0.duplicate());
            subvertex_i+=1;
    
            // Layers 1 to `frequency`-1
            // => Done only if (frequency > 1 (<=> max_proj_i > 0)).
            for layer_i in 1..layers-1 {
                let layer_points = layer_i+1;

                let unprojected_first: Vertex =  new_vertex_unprojected(&v0, &v1, layer_i, layers-1);
                let unprojected_last: Vertex =  new_vertex_unprojected(&v0, &v2, layer_i, layers-1);

                // Point 0
                let mult = projection_multipliers[subvertex_i];
                vertices.push(unprojected_first.duplicate().transform(Transformation::Scale(mult, mult, mult)).start_t(start).end_t(end));
                subvertex_i+=1;

                // Points 1 to penultimate
                for point_i in 1..layer_points-1 {
                    let mult = projection_multipliers[subvertex_i];
                    vertices.push(new_vertex_unprojected(&unprojected_first, &unprojected_last, point_i, layer_points-1).transform(Transformation::Scale(mult, mult, mult)).start_t(start).end_t(end));
                    subvertex_i+=1;
                }

                // Last point
                vertices.push(unprojected_last.duplicate().transform(Transformation::Scale(mult, mult, mult)).start_t(start).end_t(end));
                subvertex_i+=1;
            }

            // Last layer
    
            // Point 0
            vertices.push(v1.duplicate());
            subvertex_i+=1;
            
            // Points 1 to penultimate
            let layer_points = layers;
            for point_i in 1..layer_points-1 {
                let mult = projection_multipliers[subvertex_i];
                vertices.push(new_vertex_unprojected(&v1, &v2, point_i, layer_points-1).transform(Transformation::Scale(mult, mult, mult)).start_t(start).end_t(end));
                subvertex_i+=1;
            }
    
            // Last point
            vertices.push(v2);
            subvertex_i+=1;

            assert!(subpoints==subvertex_i);
        }


        // -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

        // Organizing all vertices into triangles

        let mut triangles: Vec<Vertex> = vec![];

        for layer_i in 0..layers-1 {
            let subface_red = (f32()+1.)/2.;
            let first_point = (layer_i) * (layer_i+1) / 2;
            let next_first_point = (layer_i+1) * (layer_i+2) / 2;
            // First triangle of the layer
            triangles.push(vertices[(first_point) as usize].clone().rgba(face_red, 0., 0., 1.0).recolor(Colorization::ToColor(subface_red, 0., 0., 1.0)).start_c(start).end_c(end));
            triangles.push(vertices[(next_first_point) as usize].clone().rgba(face_red, 0., 0., 1.0).recolor(Colorization::ToColor(subface_red, 0., 0., 1.0)).start_c(start).end_c(end));
            triangles.push(vertices[(next_first_point + 1) as usize].clone().rgba(face_red, 0., 0., 1.0).recolor(Colorization::ToColor(subface_red, 0., 0., 1.0)).start_c(start).end_c(end));

            // Every other pair of triangles in the layer (the 2 triangles that vertex belongs to, that are on the right of that vertex on the above pyramid vizualization)
            for point_added_i in 0..(next_first_point-first_point-1)  {
                let subface_red = (f32()+1.)/2.;
                triangles.push(vertices[(point_added_i + first_point) as usize].clone().rgba(face_red, 0., 0., 1.0).recolor(Colorization::ToColor(subface_red, 0., 0., 1.0)).start_c(start).end_c(end));
                triangles.push(vertices[(point_added_i + first_point + 1) as usize].clone().rgba(face_red, 0., 0., 1.0).recolor(Colorization::ToColor(subface_red, 0., 0., 1.0)).start_c(start).end_c(end));
                triangles.push(vertices[(point_added_i + next_first_point + 1) as usize].clone().rgba(face_red, 0., 0., 1.0).recolor(Colorization::ToColor(subface_red, 0., 0., 1.0)).start_c(start).end_c(end));

                let subface_red = (f32()+1.)/2.;
                triangles.push(vertices[(point_added_i + first_point + 1) as usize].clone().rgba(face_red, 0., 0., 1.0).recolor(Colorization::ToColor(subface_red, 0., 0., 1.0)).start_c(start).end_c(end));
                triangles.push(vertices[(point_added_i + next_first_point + 1) as usize].clone().rgba(face_red, 0., 0., 1.0).recolor(Colorization::ToColor(subface_red, 0., 0., 1.0)).start_c(start).end_c(end));
                triangles.push(vertices[(point_added_i + next_first_point + 2) as usize].clone().rgba(face_red, 0., 0., 1.0).recolor(Colorization::ToColor(subface_red, 0., 0., 1.0)).start_c(start).end_c(end));
            }
        }
        
        triangles
    }).collect();
    
    let mut out = Model::from_vertices(all_vertices);
    out.t = base_model.t;
    out
}

// TODO Remove and specify the timing of each interior frequency to 
/// Same as geodesic_sphere, but expects `base_model` to be such a sphere. The difference is that no optimizations will be made here (although there could be, but different ones).
#[allow(unused)]
fn refine_sphere(base_model: Model, frequency: u32, start: f32, end: f32) -> Model {
    assert!(
        frequency != 0 && 
        base_model.models.is_empty() &&
        !base_model.vertices.is_empty() &&
        base_model.vertices.len()%3 == 0
    );

    // The number of lines the subtriangles form (in the case of layer 0, it is a point)
    let layers = frequency+1;

    let mut sample_pos = base_model.vertices[1].position;
    for t_i in base_model.vertices[1].t.iter() {
        if let Transformation::Scale(mult, _, _) = t_i.t {
            sample_pos[0]*=mult;
            sample_pos[1]*=mult;
            sample_pos[2]*=mult;
        } else {
            println!("This code isn't meant for other transformations than scale. Be careful!");
        }
    }
    let radius = (sample_pos[0].powi(2) + sample_pos[1].powi(2) + sample_pos[2].powi(2)).sqrt();

    let subpoints = ((frequency+1) * (frequency+2) / 2) as usize;

    let all_vertices: Vec<Vertex> = base_model.vertices.into_iter().chunks(3).into_iter().flat_map(|mut v| {
        // Finding all vertices
        // The following iterates through each layer and each of its points to find the vertices and their coordinates.
        let mut vertices: Vec<Vertex> = Vec::with_capacity(subpoints);

        let v0 = v.next().unwrap();
        let v1 = v.next().unwrap();
        let v2 = v.next().unwrap();
            
        // Layer 0
        vertices.push(v0.clone());
    
        // Layers 1 to `frequency`-1
        // => Done only if (frequency > 1 (<=> max_proj_i > 0)).
        for layer_i in 1..layers-1 {
            let layer_points = layer_i+1;
            let unprojected_first: Vertex = new_vertex_unprojected_cloned(&v0, &v1, layer_i, layers-1);
            let unprojected_last: Vertex = new_vertex_unprojected_cloned(&v0, &v2, layer_i, layers-1);

            // Point 0
            vertices.push(unprojected_first.clone().scale_to_rad(radius, start, end));

            // Points 1 to penultimate
            for point_i in 1..layer_points-1 {
                let unprojected_point = new_vertex_unprojected_cloned(&unprojected_first, &unprojected_last, point_i, layer_points-1);
                vertices.push(unprojected_point.scale_to_rad(radius, start, end));
            }

            // Last point
            vertices.push(unprojected_last.clone().scale_to_rad(radius, start, end));
        }

        // Last layer
    
        // Point 0
        vertices.push(v1.clone());
        
        // Points 1 to penultimate
        let layer_points = layers;
        for point_i in 1..layer_points-1 {
            let unprojected_point = new_vertex_unprojected_cloned(&v1, &v2, point_i, layer_points-1);
            vertices.push(unprojected_point.scale_to_rad(radius, start, end));
        }
    
        // Last point
        vertices.push(v2);


        // -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

        // Organizing all vertices into triangles

        let mut triangles: Vec<Vertex> = vec![];

        for layer_i in 0..layers-1 {
            let subface_red = (f32()+1.)/2.;
            let first_point = (layer_i) * (layer_i+1) / 2;
            let next_first_point = (layer_i+1) * (layer_i+2) / 2;
            // First triangle of the layer
            triangles.push(vertices[(first_point) as usize].clone().recolor(Colorization::ToColor(subface_red, 0., 0., 1.0)).start_c(start).end_c(end));
            triangles.push(vertices[(next_first_point) as usize].clone().recolor(Colorization::ToColor(subface_red, 0., 0., 1.0)).start_c(start).end_c(end));
            triangles.push(vertices[(next_first_point + 1) as usize].clone().recolor(Colorization::ToColor(subface_red, 0., 0., 1.0)).start_c(start).end_c(end));

            // Every other pair of triangles in the layer (the 2 triangles that vertex belongs to, that are on the right of that vertex on the above pyramid vizualization)
            for point_added_i in 0..(next_first_point-first_point-1)  {
                let subface_red = (f32()+1.)/2.;
                triangles.push(vertices[(point_added_i + first_point) as usize].clone().recolor(Colorization::ToColor(subface_red, 0., 0., 1.0)).start_c(start).end_c(end));
                triangles.push(vertices[(point_added_i + first_point + 1) as usize].clone().recolor(Colorization::ToColor(subface_red, 0., 0., 1.0)).start_c(start).end_c(end));
                triangles.push(vertices[(point_added_i + next_first_point + 1) as usize].clone().recolor(Colorization::ToColor(subface_red, 0., 0., 1.0)).start_c(start).end_c(end));

                let subface_red = (f32()+1.)/2.;
                triangles.push(vertices[(point_added_i + first_point + 1) as usize].clone().recolor(Colorization::ToColor(subface_red, 0., 0., 1.0)).start_c(start).end_c(end));
                triangles.push(vertices[(point_added_i + next_first_point + 1) as usize].clone().recolor(Colorization::ToColor(subface_red, 0., 0., 1.0)).start_c(start).end_c(end));
                triangles.push(vertices[(point_added_i + next_first_point + 2) as usize].clone().recolor(Colorization::ToColor(subface_red, 0., 0., 1.0)).start_c(start).end_c(end));
            }
        }
        
        triangles
    }).collect();
    
    Model::from_vertices(all_vertices)
}

/// Returns a vertex on the line which goes from V1 to V2, at V/W.
fn new_vertex_unprojected(v1: &Vertex, v2: &Vertex, v: u32, w: u32) -> Vertex {
    assert!(v<w);
    let x = (v1.position[0] * (w-v) as f32 + v2.position[0] * v as f32) / w as f32;
    let y = (v1.position[1] * (w-v) as f32 + v2.position[1] * v as f32) / w as f32;
    let z = (v1.position[2] * (w-v) as f32 + v2.position[2] * v as f32) / w as f32;

    Vertex::new().pos(x, y, z)
}

/// Returns a vertex on the line which goes from V1 to V2, at V/W, with the same transformations and colorizations as v1.
fn new_vertex_unprojected_cloned(v1: &Vertex, v2: &Vertex, v: u32, w: u32) -> Vertex {
    assert!(v<w);
    let x = (v1.position[0] * (w-v) as f32 + v2.position[0] * v as f32) / w as f32;
    let y = (v1.position[1] * (w-v) as f32 + v2.position[1] * v as f32) / w as f32;
    let z = (v1.position[2] * (w-v) as f32 + v2.position[2] * v as f32) / w as f32;

    let out = (*v1).clone();
    out.pos(x, y, z)
}

/// The base Tetrahedron
fn tetrahedron() -> Model {
    let mult = 1./3.0f32.sqrt();
    Model::from_vertices(vec![
        Vertex::new().pos(1., 1., 1.)   ,Vertex::new().pos(-1., -1., 1.)    ,Vertex::new().pos(-1., 1., -1.)    ,
        Vertex::new().pos(1., 1., 1.)   ,Vertex::new().pos(-1., 1., -1.)    ,Vertex::new().pos(1., -1., -1.)    ,
        Vertex::new().pos(1., 1., 1.)   ,Vertex::new().pos(1., -1., -1.)    ,Vertex::new().pos(-1., -1., 1.)      ,
        Vertex::new().pos(-1., -1., 1.) ,Vertex::new().pos(-1., 1., -1.)    ,Vertex::new().pos(1., -1., -1.)    ,
    ]).transform(Transformation::Scale(mult, mult, mult)).start_t(0.).end_t(0.)
}

/// The base Octahedron
fn octahedron() -> Model {
    Model::from_vertices(vec![
        Vertex::new().pos(-1., 0., 0.)  ,Vertex::new().pos(0., 1., 0.)  ,Vertex::new().pos(0., 0., -1.) ,
        Vertex::new().pos(-1., 0., 0.)  ,Vertex::new().pos(0., 0., -1.) ,Vertex::new().pos(0., -1., 0.) ,
        Vertex::new().pos(-1., 0., 0.)  ,Vertex::new().pos(0., -1., 0.) ,Vertex::new().pos(0., 0., 1.)  ,
        Vertex::new().pos(-1., 0., 0.)  ,Vertex::new().pos(0., 0., 1.)  ,Vertex::new().pos(0., 1., 0.)  ,

        Vertex::new().pos(1., 0., 0.)   ,Vertex::new().pos(0., 1., 0.)  ,Vertex::new().pos(0., 0., -1.) ,
        Vertex::new().pos(1., 0., 0.)   ,Vertex::new().pos(0., 0., -1.) ,Vertex::new().pos(0., -1., 0.) ,
        Vertex::new().pos(1., 0., 0.)   ,Vertex::new().pos(0., -1., 0.) ,Vertex::new().pos(0., 0., 1.)  ,
        Vertex::new().pos(1., 0., 0.)   ,Vertex::new().pos(0., 0., 1.)  ,Vertex::new().pos(0., 1., 0.)  ,
    ])
}

/// The base Icosahedron
fn icosahedron() -> Model {
    let mult = 1./(PHI.powi(2)+1.).sqrt();
    Model::from_vertices(vec![
        Vertex::new().pos(0., 1., PHI)      ,Vertex::new().pos(1., PHI, 0.)     ,Vertex::new().pos(-1., PHI, 0.)    ,
        Vertex::new().pos(0., 1., PHI)      ,Vertex::new().pos(-1., PHI, 0.)    ,Vertex::new().pos(-PHI, 0., 1.)    ,
        Vertex::new().pos(0., 1., PHI)      ,Vertex::new().pos(-PHI, 0., 1.)    ,Vertex::new().pos(0., -1., PHI)    ,
        Vertex::new().pos(0., 1., PHI)      ,Vertex::new().pos(0., -1., PHI)    ,Vertex::new().pos(PHI, 0., 1.)     ,
        Vertex::new().pos(0., 1., PHI)      ,Vertex::new().pos(PHI, 0., 1.)     ,Vertex::new().pos(1., PHI, 0.)     ,

        Vertex::new().pos(0., -1., PHI)     ,Vertex::new().pos(-PHI, 0., 1.)    ,Vertex::new().pos(-1., -PHI, 0.)   ,
        Vertex::new().pos(0., -1., PHI)     ,Vertex::new().pos(-1., -PHI, 0.)   ,Vertex::new().pos(1., -PHI, 0.)    ,
        Vertex::new().pos(0., -1., PHI)     ,Vertex::new().pos(1., -PHI, 0.)    ,Vertex::new().pos(PHI, 0., 1.)     ,
        Vertex::new().pos(0., -1., -PHI)    ,Vertex::new().pos(0., 1., -PHI)    ,Vertex::new().pos(PHI, 0., -1.)    ,
        Vertex::new().pos(0., -1., -PHI)    ,Vertex::new().pos(PHI, 0., -1.)    ,Vertex::new().pos(1., -PHI, 0.)    ,

        Vertex::new().pos(0., -1., -PHI)    ,Vertex::new().pos(1., -PHI, 0.)    ,Vertex::new().pos(-1., -PHI, 0.)   ,
        Vertex::new().pos(0., -1., -PHI)    ,Vertex::new().pos(-1., -PHI, 0.)   ,Vertex::new().pos(-PHI, 0., -1.)   ,
        Vertex::new().pos(0., -1., -PHI)    ,Vertex::new().pos(-PHI, 0., -1.)   ,Vertex::new().pos(0., 1., -PHI)    ,
        Vertex::new().pos(0., 1., -PHI)     ,Vertex::new().pos(-PHI, 0., -1.)   ,Vertex::new().pos(-1., PHI, 0.)    ,
        Vertex::new().pos(0., 1., -PHI)     ,Vertex::new().pos(-1., PHI, 0.)    ,Vertex::new().pos(1., PHI, 0.)     ,

        Vertex::new().pos(0., 1., -PHI)     ,Vertex::new().pos(1., PHI, 0.)     ,Vertex::new().pos(PHI, 0., -1.)    ,
        Vertex::new().pos(PHI, 0., 1.)      ,Vertex::new().pos(1., -PHI, 0.)    ,Vertex::new().pos(PHI, 0., -1.)    ,
        Vertex::new().pos(PHI, 0., 1.)      ,Vertex::new().pos(PHI, 0., -1.)    ,Vertex::new().pos(1., PHI, 0.)     ,
        Vertex::new().pos(-PHI, 0., 1.)     ,Vertex::new().pos(-1., PHI, 0.)    ,Vertex::new().pos(-PHI, 0., -1.)   ,
        Vertex::new().pos(-PHI, 0., 1.)     ,Vertex::new().pos(-PHI, 0., -1.)   ,Vertex::new().pos(-1., -PHI, 0.)   ,
    ]).transform(Transformation::Scale(mult, mult, mult)).start_t(0.).end_t(0.)
}


trait Project<T> {
    fn scale_to_rad(self: Self, rad: f32, start: f32, end: f32) -> Self;
}

impl Project<Vertex> for Vertex {
    // Rotates linearly by pi around a y axis with an X offset from START to END CLOCKWISE
    fn scale_to_rad(self: Self, rad: f32, start: f32, end: f32) -> Self {
        let [mut x, mut y, mut z, _] = self.position;
        self.t.iter().for_each(|t| {
            if let Transformation::Scale(factor, _, _) = t.t {
                x*=factor;
                y*=factor;
                z*=factor;
            }
        });
        let mult = rad/(x.powi(2) + y.powi(2) + z.powi(2)).sqrt();
        self
            .transform(Transformation::Scale(mult, mult, mult)).start_t(start).end_t(end)
    }
}