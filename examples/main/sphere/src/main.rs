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
    while v.show() {
        v.reset(get())
    };
}
fn get() -> Input {
    // // Geodesic division of Tetrahedron, Octahedron and Icosahedron

    // Frequencies can generally go over 100, it's just a bit of loading after that.
    // let sphere1 = geodesic_sphere(tetrahedron(), 20, 2., 5.); 
    // let sphere2 = geodesic_sphere(octahedron(), 20, 6., 9.);
    // let sphere3 = geodesic_sphere(icosahedron(), 20, 10., 13.);

    // println!("Sphere 1: Tetrahedron base");
    // println!("- {} triangles.", sphere1.vertices.len()/3);
    // println!("- {} vertices.", (sphere1.vertices.len()+4*3)/6); // Add the number of lacking vertices for hexagonal (6) division (here: 3 per tetrahedron corner, for the 4 corners. 12 anyway). The returned geodesic sphere's vertices are all present 6 times, except the vertices of the base platonic solid.

    // println!("Sphere 2: Octahedron base");
    // println!("- {} triangles.", sphere2.vertices.len()/3);
    // println!("- {} vertices.", (sphere2.vertices.len()+6*2)/6);

    // println!("Sphere 3: Icosahedron base");
    // println!("- {} triangles.", sphere3.vertices.len()/3);
    // println!("- {} vertices.", (sphere3.vertices.len()+12*1)/6);

    // Input {
    //     meta: MetaInput {
    //         bg: [0.3, 0.3, 0.3, 0.3],
    //         start: 0.0,
    //         end: 14.0,
    //     },
    //     m: vec![
    //         sphere1
    //             .transform(Transformation::Scale(0.5, 0.5, 0.5)).start_t(0.0).end_t(0.0)
    //             .transform(Transformation::Scale(2., 2., 2.)).start_t(1.0).end_t(2.0)
    //             .transform(Transformation::Scale(0.1, 0.1, 0.1)).start_t(5.0).end_t(6.0)
    //             .transform(Transformation::RotateY(2.*PI)).start_t(2.0).end_t(5.0),
    //         sphere2
    //             .transform(Transformation::Scale(0.1, 0.1, 0.1)).start_t(0.0).end_t(0.0)
    //             .transform(Transformation::Scale(10., 10., 10.)).start_t(5.0).end_t(6.0)
    //             .transform(Transformation::Scale(0.1, 0.1, 0.1)).start_t(9.0).end_t(10.0)
    //             .transform(Transformation::RotateY(2.*PI)).start_t(6.0).end_t(9.0),
    //         sphere3
    //             .transform(Transformation::Scale(0.1, 0.1, 0.1)).start_t(0.0).end_t(0.0)
    //             .transform(Transformation::Scale(10., 10., 10.)).start_t(9.0).end_t(10.0)
    //             .transform(Transformation::Scale(0.2, 0.2, 0.2)).start_t(13.0).end_t(14.0)
    //             .transform(Transformation::RotateY(2.*PI)).start_t(10.0).end_t(13.0),
    //     ],
    //     v: View::new().transform(Transformation::Lookat(0., 0., -3., 0., 0., 0., 0., -1., 0.)),
    //     p: Projection::new().transform(Transformation::Perspective(-0.1, 0.1, -0.1, 0.1, 0.2, 100.)),
    // }




    // // Triangle / Arrow / Sphere scene.
    // 
    // unsafe {
    //     D_TRANSFORMATION_START_TIME = -1.;
    //     D_TRANSFORMATION_END_TIME = -1.;
    //     D_COLORIZATION_START_TIME = -1.;
    //     D_COLORIZATION_END_TIME = -1.;
    // }
    // 
    // // Frequencies can generally go over 100, it's just a bit of loading after that.
    // let triangle = Model::from_vertices(vec![
    //     Vertex::new().pos(-2.0, -0.25, 0.0).rgba(1.0, 0.0, 0.0, 1.0)
    //     .recolor(Colorization::ToColor(1., f32(), f32(), 1.)).start_c(0.).end_c(2.)
    //     .recolor(Colorization::ToColor(1., f32(), f32(), 1.)).start_c(2.).end_c(4.)
    //     .recolor(Colorization::ToColor(1., f32(), f32(), 1.)).start_c(4.).end_c(6.)
    //     .recolor(Colorization::ToColor(1., f32(), f32(), 1.)).start_c(6.).end_c(8.)
    //     .recolor(Colorization::ToColor(1., f32(), f32(), 1.)).start_c(8.).end_c(10.),
    //     Vertex::new().pos(-1.0, -0.25, 0.0).rgba(1.0, 0.0, 0.0, 1.0)
    //     .recolor(Colorization::ToColor(1., f32(), f32(), 1.)).start_c(0.).end_c(2.)
    //     .recolor(Colorization::ToColor(1., f32(), f32(), 1.)).start_c(2.).end_c(4.)
    //     .recolor(Colorization::ToColor(1., f32(), f32(), 1.)).start_c(4.).end_c(6.)
    //     .recolor(Colorization::ToColor(1., f32(), f32(), 1.)).start_c(6.).end_c(8.)
    //     .recolor(Colorization::ToColor(1., f32(), f32(), 1.)).start_c(8.).end_c(10.),
    //     Vertex::new().pos(-1.5, 0.5, 0.0).rgba(1.0, 0.0, 0.0, 1.0)
    //     .recolor(Colorization::ToColor(1., f32(), f32(), 1.)).start_c(0.).end_c(2.)
    //     .recolor(Colorization::ToColor(1., f32(), f32(), 1.)).start_c(2.).end_c(4.)
    //     .recolor(Colorization::ToColor(1., f32(), f32(), 1.)).start_c(4.).end_c(6.)
    //     .recolor(Colorization::ToColor(1., f32(), f32(), 1.)).start_c(6.).end_c(8.)
    //     .recolor(Colorization::ToColor(1., f32(), f32(), 1.)).start_c(8.).end_c(10.),
    // ])
    // .transform(Transformation::Translate(0.0, -0.125, 0.0));
        // 
    // let arrow = Model::from_vertices(vec![
    //     Vertex::new().pos(-0.35, -0.05, 0.0).rgba(1.0, 0.0, 0.0, 1.0),
    //     Vertex::new().pos(-0.35,  0.05, 0.0).rgba(1.0, 0.0, 0.0, 1.0),
    //     Vertex::new().pos( 0.25, -0.05, 0.0).rgba(1.0, 0.0, 0.0, 1.0),
    //     Vertex::new().pos(-0.35,  0.05, 0.0).rgba(1.0, 0.0, 0.0, 1.0),
    //     Vertex::new().pos( 0.25, -0.05, 0.0).rgba(1.0, 0.0, 0.0, 1.0),
    //     Vertex::new().pos( 0.25,  0.05, 0.0).rgba(1.0, 0.0, 0.0, 1.0),
    //     
    //     Vertex::new().pos( 0.35,  0.00, 0.0).rgba(1.0, 0.0, 0.0, 1.0),
    //     Vertex::new().pos( 0.25,  0.00, 0.0).rgba(1.0, 0.0, 0.0, 1.0),
    //     Vertex::new().pos( 0.20,  0.15, 0.0).rgba(1.0, 0.0, 0.0, 1.0),
    //     Vertex::new().pos( 0.25,  0.00, 0.0).rgba(1.0, 0.0, 0.0, 1.0),
    //     Vertex::new().pos( 0.20,  0.15, 0.0).rgba(1.0, 0.0, 0.0, 1.0),
    //     Vertex::new().pos( 0.10,  0.15, 0.0).rgba(1.0, 0.0, 0.0, 1.0),
    //     
    //     Vertex::new().pos( 0.35,  0.00, 0.0).rgba(1.0, 0.0, 0.0, 1.0),
    //     Vertex::new().pos( 0.25,  0.00, 0.0).rgba(1.0, 0.0, 0.0, 1.0),
    //     Vertex::new().pos( 0.20, -0.15, 0.0).rgba(1.0, 0.0, 0.0, 1.0),
    //     Vertex::new().pos( 0.25,  0.00, 0.0).rgba(1.0, 0.0, 0.0, 1.0),
    //     Vertex::new().pos( 0.20, -0.15, 0.0).rgba(1.0, 0.0, 0.0, 1.0),
    //     Vertex::new().pos( 0.10, -0.15, 0.0).rgba(1.0, 0.0, 0.0, 1.0),
    // ]);
        // 
    // let sphere = geodesic_sphere(icosahedron(), 50, -1., -1.)
    // .recolor(Colorization::ToColor(1., 0., 0., 1.))
    // .recolor(Colorization::ToColor(1., f32(), f32(), 1.)).start_c(0.).end_c(1.)
    // .recolor(Colorization::ToColor(1., f32(), f32(), 1.)).start_c(1.).end_c(2.)
    // .recolor(Colorization::ToColor(1., f32(), f32(), 1.)).start_c(2.).end_c(3.)
    // .recolor(Colorization::ToColor(1., f32(), f32(), 1.)).start_c(3.).end_c(4.)
    // .recolor(Colorization::ToColor(1., f32(), f32(), 1.)).start_c(4.).end_c(5.)
    // .recolor(Colorization::ToColor(1., f32(), f32(), 1.)).start_c(5.).end_c(6.)
    // .recolor(Colorization::ToColor(1., f32(), f32(), 1.)).start_c(6.).end_c(7.)
    // .recolor(Colorization::ToColor(1., f32(), f32(), 1.)).start_c(7.).end_c(8.)
    // .recolor(Colorization::ToColor(1., f32(), f32(), 1.)).start_c(8.).end_c(9.)
    // .recolor(Colorization::ToColor(1., f32(), f32(), 1.)).start_c(9.).end_c(10.)
    // .transform(Transformation::Scale(0.5, 0.5, 0.5))
    // .transform(Transformation::Translate(1.5, 0.0, 0.0));
        // 
    // Input {
    //     meta: MetaInput {
    //         bg: [0.1, 0.1, 0.1, 0.1],
    //         start: -1.0,
    //         end: 11.0,
    //     },
    //     m: vec![
    //         triangle
    //         .transform(Transformation::Scale(0.0001, 0.0001, 0.0001)).transform(Transformation::Scale(10000.0, 10000.0, 10000.0)).start_t(0.).end_t(1.).evolution_t(Evolution::FastIn).transform(Transformation::RotateY(-3.0 * PI)).start_t(0.).end_t(10.).evolution_t(Evolution::FastMiddle).transform(Transformation::Scale(0.0001, 0.0001, 0.0001)).start_t(9.).end_t(10.).evolution_t(Evolution::FastOut),
    //         arrow
    //         .transform(Transformation::Scale(0.0001, 0.0001, 0.0001)).transform(Transformation::Scale(10000.0, 10000.0, 10000.0)).start_t(0.).end_t(1.).evolution_t(Evolution::FastIn).transform(Transformation::RotateY(-3.0 * PI)).start_t(0.).end_t(10.).evolution_t(Evolution::FastMiddle).transform(Transformation::Scale(0.0001, 0.0001, 0.0001)).start_t(9.).end_t(10.).evolution_t(Evolution::FastOut),
    //         sphere
    //         .transform(Transformation::Scale(0.0001, 0.0001, 0.0001)).transform(Transformation::Scale(10000.0, 10000.0, 10000.0)).start_t(0.).end_t(1.).evolution_t(Evolution::FastIn).transform(Transformation::RotateY(-3.0 * PI)).start_t(0.).end_t(10.).evolution_t(Evolution::FastMiddle).transform(Transformation::Scale(0.0001, 0.0001, 0.0001)).start_t(9.).end_t(10.).evolution_t(Evolution::FastOut),
    //     ],
    //     v: View::new().transform(Transformation::Lookat(0., 0., -3., 0., 0., 0., 0., -1., 0.)),
    //     p: Projection::new().transform(Transformation::Perspective(-0.1, 0.1, -0.1, 0.1, 0.2, 100.)),
    // }




    // Display Icosahedron, divided with a frequency of 8.

    unsafe {
        D_TRANSFORMATION_START_TIME = 0.;
        D_TRANSFORMATION_END_TIME = 0.;
    }

    // let sphere1 = geodesic_sphere(icosahedron(), 1, 0., 0.);
    // let sphere2 = geodesic_sphere(icosahedron(), 2, 0., 0.);
    // let sphere3 = geodesic_sphere(icosahedron(), 3, 0., 0.);
    // let sphere4 = geodesic_sphere(icosahedron(), 4, 0., 0.);
    // let sphere5 = geodesic_sphere(icosahedron(), 5, 0., 0.);
    // let sphere6 = geodesic_sphere(icosahedron(), 6, 0., 0.);
    // let sphere7 = geodesic_sphere(icosahedron(), 7, 0., 0.);
    let sphere8 = geodesic_sphere(icosahedron(), 8, 0., 5.);
    // let sphere9 = geodesic_sphere(icosahedron(), 9, 0., 0.);
    // let sphere10 = geodesic_sphere(icosahedron(), 10, 0., 0.);
    // let sphere11 = geodesic_sphere(icosahedron(), 11, 0., 0.);
    // let sphere12 = geodesic_sphere(icosahedron(), 12, 0., 0.);
    // let sphere13 = geodesic_sphere(icosahedron(), 13, 0., 0.);
    // let sphere14 = geodesic_sphere(icosahedron(), 14, 0., 0.);
    // let sphere15 = geodesic_sphere(icosahedron(), 15, 0., 0.);
    // let sphere16 = geodesic_sphere(icosahedron(), 16, 0., 0.);
    // let sphere17 = geodesic_sphere(icosahedron(), 17, 0., 0.);
    // let sphere18 = geodesic_sphere(icosahedron(), 18, 0., 0.);
    // let sphere19 = geodesic_sphere(icosahedron(), 19, 0., 0.);

    Input {
        meta: MetaInput {
            bg: [0.1, 0.1, 0.1, 0.1],
            start: 0.0,
            end: 10.0,
        },
        m: vec![
            // sphere1
            //     .transform(Transformation::Scale(0.5, 0.5, 0.5)).start_t(0.0).end_t(0.0)
            //     .transform(Transformation::Scale(2., 2., 2.)).start_t(1.0).end_t(2.0)
            //     .transform(Transformation::Scale(0.1, 0.1, 0.1)).start_t(5.0).end_t(6.0)
            //     .transform(Transformation::RotateY(2.*PI)).start_t(2.0).end_t(5.0),
            // sphere2
            //     .transform(Transformation::Scale(0.1, 0.1, 0.1)).start_t(0.0).end_t(0.0)
            //     .transform(Transformation::Scale(10., 10., 10.)).start_t(5.0).end_t(6.0)
            //     .transform(Transformation::Scale(0.1, 0.1, 0.1)).start_t(9.0).end_t(10.0)
            //     .transform(Transformation::RotateY(2.*PI)).start_t(6.0).end_t(9.0),
            // sphere3
            //     .transform(Transformation::Scale(0.1, 0.1, 0.1)).start_t(0.0).end_t(0.0)
            //     .transform(Transformation::Scale(10., 10., 10.)).start_t(9.0).end_t(10.0)
            //     .transform(Transformation::Scale(0.2, 0.2, 0.2)).start_t(13.0).end_t(14.0)
            //     .transform(Transformation::RotateY(2.*PI)).start_t(10.0).end_t(13.0),

            // sphere1,
            // sphere2,
            // sphere3,
            // sphere4,
            // sphere5,
            // sphere6,
            // sphere7,
            sphere8.transform(Transformation::RotateY(PI * 4.0)).start_t(0.).end_t(10.0).evolution_t(vera::Evolution::FastMiddle),
            // sphere9,
            // sphere10,
            // sphere11,
            // sphere12,
            // sphere13,
            // sphere14,
            // sphere15,
            // sphere16,
            // sphere17,
            // sphere18,
            // sphere19,
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
    let (
        models,
        vertices,
        t,
    ) = base_model.own_fields();

    assert!(
        frequency != 0 && 
        models.is_empty() &&
        !vertices.is_empty() &&
        vertices.len()%3 == 0
    );

    // The number of lines the subtriangles form (in the case of layer 0, it is a point)
    let layers = frequency+1;

    let sample_pos = vertices[0].read_position();
    let radius: f32 = (sample_pos[0].powi(2) + sample_pos[1].powi(2) + sample_pos[2].powi(2)).sqrt();

    let subpoints = ((frequency+1) * (frequency+2) / 2) as usize;
    let mut projection_multipliers: Vec<f32> = Vec::with_capacity(subpoints);

    let all_vertices: Vec<Vertex> = vertices.into_iter().chunks(3).into_iter().enumerate().flat_map(|(base_triangle_i, mut v)| {
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
            vertices.push(v0.clone().transform(Transformation::Scale(1., 1., 1.)).start_t(start).end_t(end));
            projection_multipliers.push(1.0);
    
            // Layers 1 to `frequency`-1
            // => Done only if (frequency > 1 (<=> max_proj_i > 0)).
            for layer_i in 1..layers-1 {
                let layer_points = layer_i+1;
                let unprojected_first: Vertex = new_vertex_unprojected(&v0, &v1, layer_i, layers-1);
                let unprojected_last: Vertex = new_vertex_unprojected(&v0, &v2, layer_i, layers-1);
                let new_mult = radius/(unprojected_first.read_position()[0].powi(2) + unprojected_first.read_position()[1].powi(2) + unprojected_first.read_position()[2].powi(2)).sqrt();

                // Point 0
                vertices.push(unprojected_first.duplicate().transform(Transformation::Scale(new_mult, new_mult, new_mult)).start_t(start).end_t(end));
                projection_multipliers.push(new_mult);

                // Points 1 to penultimate
                for point_i in 1..layer_points-1 {
                    let unprojected_point = new_vertex_unprojected(&unprojected_first, &unprojected_last, point_i, layer_points-1);
                    let new_mult = radius/(unprojected_point.read_position()[0].powi(2) + unprojected_point.read_position()[1].powi(2) + unprojected_point.read_position()[2].powi(2)).sqrt();
                    vertices.push(unprojected_point.transform(Transformation::Scale(new_mult, new_mult, new_mult)).start_t(start).end_t(end));
                    projection_multipliers.push(new_mult);
                }

                // Last point
                vertices.push(unprojected_last.duplicate().transform(Transformation::Scale(new_mult, new_mult, new_mult)).start_t(start).end_t(end));
                projection_multipliers.push(new_mult);
            }

            // Last layer
    
            // Point 0
            vertices.push(v1.duplicate().transform(Transformation::Scale(1., 1., 1.)).start_t(start).end_t(end));
            projection_multipliers.push(1.0);
            
            // Points 1 to penultimate
            let layer_points = layers;
            for point_i in 1..layer_points-1 {
                let unprojected_point = new_vertex_unprojected(&v1, &v2, point_i, layer_points-1);
                let new_mult = radius/(unprojected_point.read_position()[0].powi(2) + unprojected_point.read_position()[1].powi(2) + unprojected_point.read_position()[2].powi(2)).sqrt();
                vertices.push(unprojected_point.transform(Transformation::Scale(new_mult, new_mult, new_mult)).start_t(start).end_t(end));
                projection_multipliers.push(new_mult);
            }
    
            // Last point
            vertices.push(v2.transform(Transformation::Scale(1., 1., 1.)).start_t(start).end_t(end));
            assert!(vertices.len()>0);
            projection_multipliers.push(1.0);

            // indices of the vertices of the triangle (only one OR 1 in 3) that will be the biggest after projection.
            if frequency>0 {
                let most_projected_triangle_layer_i: usize = (frequency as isize/3*2-1 + frequency as isize%3) as usize;
                let indices: (usize, usize, usize) = match most_projected_triangle_layer_i%2 {
                    0 => (
                        most_projected_triangle_layer_i*(most_projected_triangle_layer_i+2)/2,
                        most_projected_triangle_layer_i*(most_projected_triangle_layer_i+2)/2 + most_projected_triangle_layer_i+1,
                        most_projected_triangle_layer_i*(most_projected_triangle_layer_i+2)/2 + most_projected_triangle_layer_i+2,
                    ),
                    _ => (
                        most_projected_triangle_layer_i*(most_projected_triangle_layer_i+2)/2,
                        most_projected_triangle_layer_i*(most_projected_triangle_layer_i+2)/2 + 1,
                        most_projected_triangle_layer_i*(most_projected_triangle_layer_i+2)/2 + most_projected_triangle_layer_i+2,
                    ),
                };
                print_ratio(&vertices[indices.0], &vertices[indices.1], &vertices[indices.2], frequency);
            }
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
    out.set_t(t);
    out
}

/// Prints the ratio between the lowest point on the geodesic sphere and the sphere radius
fn print_ratio(v1: &Vertex, v2: &Vertex, v3: &Vertex, freq: u32) {
    let end_pos_1 = end_position(v1);
    let end_pos_2 = end_position(v2);
    let end_pos_3 = end_position(v3);

    let max_dist = (end_pos_1[0].powi(2)+end_pos_1[1].powi(2)+end_pos_1[2].powi(2)).sqrt();
    let min_dist = projected_origin_distance(end_pos_1, end_pos_2, end_pos_3);

    assert!(min_dist<max_dist);

    println!("Maximum projection ratio for frequency {}: {:02}", freq, min_dist/max_dist);
}

/// Finds the end position of a vertex which only transformation is a scale, uniform between the axis.
fn end_position(v: &Vertex) -> [f32; 3] {
    if let Transformation::Scale(factor, _, _) = *v.read_tf()[0].read_t() {
    [
        v.read_position()[0] * factor,
        v.read_position()[1] * factor,
        v.read_position()[2] * factor
    ]
    } else { [0., 0., 0.] }
}

fn projected_origin_distance(v1: [f32; 3], v2: [f32; 3], v3: [f32; 3]) -> f32 {
    let vec_ab = [v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]];
    let vec_ac = [v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]];

    let normal_vector = [
        vec_ab[1] * vec_ac[2] - vec_ac[1] * vec_ab[2],
        vec_ab[2] * vec_ac[0] - vec_ac[2] * vec_ab[0],
        vec_ab[0] * vec_ac[1] - vec_ac[0] * vec_ab[1],
    ];
    
    let distance = f32::abs(
        normal_vector[0] * v1[0] + normal_vector[1] * v1[1] + normal_vector[2] * v1[2]
    ) / (normal_vector[0].powi(2) + normal_vector[1].powi(2) + normal_vector[2].powi(2)).sqrt();

    distance
}

// TODO Remove and specify the timing of each interior frequency to `geodesic_sphere`.
/// Same as geodesic_sphere, but expects `base_model` to be any geodesic sphere. The difference is that no optimizations will be made here (although there could be, but different ones).
#[allow(unused)]
fn refine_sphere(base_model: Model, frequency: u32, start: f32, end: f32) -> Model {
    let (
        models,
        vertices,
        t,
    ) = base_model.own_fields();

    assert!(
        frequency != 0 && 
        models.is_empty() &&
        !vertices.is_empty() &&
        vertices.len()%3 == 0
    );

    // The number of lines the subtriangles form (in the case of layer 0, it is a point)
    let layers = frequency+1;

    let mut sample_pos = vertices[1].read_position().clone();
    for t_i in vertices[1].read_tf().iter() {
        if let Transformation::Scale(mult, _, _) = *t[0].read_t() {
            sample_pos[0]*=mult;
            sample_pos[1]*=mult;
            sample_pos[2]*=mult;
        } else {
            println!("This code isn't meant for other transformations than scale. Be careful!");
        }
    }
    let radius = (sample_pos[0].powi(2) + sample_pos[1].powi(2) + sample_pos[2].powi(2)).sqrt();

    let subpoints = ((frequency+1) * (frequency+2) / 2) as usize;

    let all_vertices: Vec<Vertex> = vertices.into_iter().chunks(3).into_iter().flat_map(|mut v| {
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
    let x = (v1.read_position()[0] * (w-v) as f32 + v2.read_position()[0] * v as f32) / w as f32;
    let y = (v1.read_position()[1] * (w-v) as f32 + v2.read_position()[1] * v as f32) / w as f32;
    let z = (v1.read_position()[2] * (w-v) as f32 + v2.read_position()[2] * v as f32) / w as f32;

    Vertex::new().pos(x, y, z)
}

/// Returns a vertex on the line which goes from V1 to V2, at V/W, with the same transformations and colorizations as v1.
fn new_vertex_unprojected_cloned(v1: &Vertex, v2: &Vertex, v: u32, w: u32) -> Vertex {
    assert!(v<w);
    let x = (v1.read_position()[0] * (w-v) as f32 + v2.read_position()[0] * v as f32) / w as f32;
    let y = (v1.read_position()[1] * (w-v) as f32 + v2.read_position()[1] * v as f32) / w as f32;
    let z = (v1.read_position()[2] * (w-v) as f32 + v2.read_position()[2] * v as f32) / w as f32;

    let out = (*v1).clone();
    out.pos(x, y, z)
}

#[allow(unused)]
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

#[allow(unused)]
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
        let [mut x, mut y, mut z, _] = self.read_position();
        self.read_tf().iter().for_each(|t| {
            if let Transformation::Scale(factor, _, _) = *t.read_t() {
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