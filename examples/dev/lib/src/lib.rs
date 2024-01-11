use std::f32::consts::PI;

use fastrand;

use vera::{
    Input, View, Projection, Model, Vertex, Transformation, MetaInput, Evolution, D_TRANSFORMATION_START_TIME, D_TRANSFORMATION_END_TIME, D_TRANSFORMATION_SPEED_EVOLUTION,
};

#[no_mangle]
fn get() -> Input {

    // Platonic life animation

    unsafe {
        D_TRANSFORMATION_START_TIME = -100.;
        D_TRANSFORMATION_END_TIME = -100.;
        D_TRANSFORMATION_SPEED_EVOLUTION = Evolution::FastMiddle;
    }

    let tetrahedron = Model::from_models(vec![
        polygon(3, [false, false, true])
            .transform(Transformation::RotateZ(PI/2.))
            .transform(Transformation::Translate(0., 1., 0.))
            .transform(Transformation::RotateX(0.3398)),
        polygon(3, [false, false, true])
            .transform(Transformation::RotateZ(PI/2.))
            .transform(Transformation::Translate(0., 1., 0.))
            .transform(Transformation::RotateX(0.3398))
            .transform(Transformation::RotateY(PI/1.5)),
        polygon(3, [false, false, true])
            .transform(Transformation::RotateZ(PI/2.))
            .transform(Transformation::Translate(0., 1., 0.))
            .transform(Transformation::RotateX(0.3398))
            .transform(Transformation::RotateY(-PI/1.5)),
        polygon(3, [false, false, true])
            .transform(Transformation::RotateZ(PI/2.))
            .transform(Transformation::RotateX(PI/2.))
            .transform(Transformation::Translate(0., 2.0f32.sqrt(), 0.)),
    ])
        .transform(Transformation::Translate(0., -1., 0.))
        
        .transform(Transformation::Scale(0.1, 0.1, 0.1))
        .transform(Transformation::Scale(10., 10., 10.)).start_t(1.).end_t(2.5)
        .transform(Transformation::Scale(1.2, 1.2, 1.2)).start_t(71.74).end_t(72.30).evolution_t(Evolution::Linear)

        .transform(Transformation::RotateY(4.*PI)).start_t(2.).end_t(7.)
        .transform(Transformation::RotateX(-2.*PI)).start_t(2.).end_t(7.)
        .transform(Transformation::RotateY(4.*PI)).start_t(7.).end_t(12.)
        .transform(Transformation::RotateX(-2.*PI)).start_t(7.).end_t(12.)
        .transform(Transformation::RotateY(4.*PI)).start_t(12.).end_t(17.)
        .transform(Transformation::RotateX(-2.*PI)).start_t(12.).end_t(17.)
        .transform(Transformation::RotateY(4.*PI)).start_t(17.).end_t(22.)
        .transform(Transformation::RotateX(-2.*PI)).start_t(17.).end_t(22.)
        .transform(Transformation::RotateY(4.*PI)).start_t(22.).end_t(27.)
        .transform(Transformation::RotateX(-2.*PI)).start_t(22.).end_t(27.)
        .transform(Transformation::RotateY(4.*PI)).start_t(27.).end_t(32.)
        .transform(Transformation::RotateX(-2.*PI)).start_t(27.).end_t(32.)
        .transform(Transformation::RotateY(4.*PI)).start_t(32.).end_t(37.)
        .transform(Transformation::RotateX(-2.*PI)).start_t(32.).end_t(37.)
        .transform(Transformation::RotateY(4.*PI)).start_t(37.).end_t(43.)
        .transform(Transformation::RotateX(-2.*PI)).start_t(37.).end_t(43.)

        .transform(Transformation::Translate(0., 3.0, 0.)).start_t(44.).end_t(55.)
        .transform(Transformation::RotateX(10.*PI)).start_t(44.).end_t(55.)
        .transform(Transformation::Translate(0., -3.0, 0.)).start_t(55.).end_t(58.)
        .transform(Transformation::RotateX(-10.*PI)).start_t(55.).end_t(58.)

        .transform(Transformation::Translate(-8., 0., 0.))

        .rotound(-6., 60.39, 60.71, true)
        .rotound(-2., 60.71, 61.03, false)
        .rotound(2., 61.03, 61.35, true)
        .rotound(6., 61.35, 61.67, false)

        .rotound(6., 64.06, 64.38, false)
        .rotound(2., 64.38, 64.80, true)
        .rotound(-2., 64.80, 65.12, false)
        .rotound(-6., 65.12, 65.44, true)

        .transform(Transformation::Translate(0., -1., 0.)).start_t(67.05).end_t(67.33).evolution_t(Evolution::FastIn)
        .transform(Transformation::Translate(0., 1., 0.)).start_t(68.17).end_t(68.45).evolution_t(Evolution::FastIn)

        .transform(Transformation::Translate(0., 1., 0.)).start_t(70.90).end_t(71.18).evolution_t(Evolution::FastIn)
        .transform(Transformation::Translate(0., -5., 0.)).start_t(72.02).end_t(72.30).evolution_t(Evolution::FastIn)
        .transform(Transformation::Translate(2., 0., 0.)).start_t(72.02).end_t(72.30).evolution_t(Evolution::FastOut)
        .transform(Transformation::Translate(0., 2., 0.)).start_t(72.30).end_t(72.58).evolution_t(Evolution::FastOut)
        .transform(Transformation::Translate(2., 0., 0.)).start_t(72.30).end_t(72.58).evolution_t(Evolution::FastIn)
        .transform(Transformation::Translate(0., 7., 0.)).start_t(72.58).end_t(72.86).evolution_t(Evolution::FastIn)
        .transform(Transformation::Translate(2., 0., 0.)).start_t(72.58).end_t(72.86).evolution_t(Evolution::FastOut)
        .transform(Transformation::Translate(0., -5., 0.)).start_t(72.86).end_t(73.14).evolution_t(Evolution::FastOut)
        .transform(Transformation::Translate(2., 0., 0.)).start_t(72.86).end_t(73.14).evolution_t(Evolution::FastIn)
    ;

    let cube = Model::from_models(vec![
        polygon(4, [false, true, true])
            .transform(Transformation::RotateX(PI/2.))
            .transform(Transformation::Translate(0.0, 2.0f32.sqrt()/2., 0.))
            ,
        polygon(4, [false, true, true])
            .transform(Transformation::RotateX(PI/2.))
            .transform(Transformation::Translate(0.0, -2.0f32.sqrt()/2., 0.))
            ,
        polygon(4, [false, true, true])
            .transform(Transformation::RotateZ(PI/4.))
            .transform(Transformation::RotateY(PI/4.))
            .transform(Transformation::Translate(0.5, 0., 0.5))
            ,
        polygon(4, [false, true, true])
            .transform(Transformation::RotateZ(PI/4.))
            .transform(Transformation::RotateY(PI/4.))
            .transform(Transformation::Translate(-0.5, 0., -0.5))
            ,
        polygon(4, [false, true, true])
            .transform(Transformation::RotateZ(PI/4.))
            .transform(Transformation::RotateY(-PI/4.))
            .transform(Transformation::Translate(0.5, 0., -0.5))
            ,
        polygon(4, [false, true, true])
            .transform(Transformation::RotateZ(PI/4.))
            .transform(Transformation::RotateY(-PI/4.))
            .transform(Transformation::Translate(-0.5, 0., 0.5))
            ,
    ])
        
        .transform(Transformation::Scale(0.1, 0.1, 0.1))
        .transform(Transformation::Scale(10., 10., 10.)).start_t(8.25).end_t(9.75)
        .transform(Transformation::RotateY(PI/4.)).start_t(72.58).end_t(72.86).evolution_t(Evolution::FastIn)

        .transform(Transformation::RotateY(4.*PI)).start_t(9.25).end_t(14.25)
        .transform(Transformation::RotateX(-2.*PI)).start_t(9.25).end_t(14.25)
        .transform(Transformation::RotateY(4.*PI)).start_t(14.25).end_t(19.25)
        .transform(Transformation::RotateX(-2.*PI)).start_t(14.25).end_t(19.25)
        .transform(Transformation::RotateY(4.*PI)).start_t(19.25).end_t(24.25)
        .transform(Transformation::RotateX(-2.*PI)).start_t(19.25).end_t(24.25)
        .transform(Transformation::RotateY(4.*PI)).start_t(24.25).end_t(29.25)
        .transform(Transformation::RotateX(-2.*PI)).start_t(24.25).end_t(29.25)
        .transform(Transformation::RotateY(4.*PI)).start_t(29.25).end_t(34.25)
        .transform(Transformation::RotateX(-2.*PI)).start_t(29.25).end_t(34.25)
        .transform(Transformation::RotateY(4.*PI)).start_t(34.25).end_t(39.25)
        .transform(Transformation::RotateX(-2.*PI)).start_t(34.25).end_t(39.25)
        .transform(Transformation::RotateY(4.*PI)).start_t(39.25).end_t(43.)
        .transform(Transformation::RotateX(-2.*PI)).start_t(39.25).end_t(43.)

        .transform(Transformation::Translate(0., 3.0, 0.)).start_t(44.).end_t(55.)
        .transform(Transformation::RotateX(8.*PI)).start_t(44.).end_t(55.)
        .transform(Transformation::Translate(0., -3.0, 0.)).start_t(55.).end_t(58.)
        .transform(Transformation::RotateX(-8.*PI)).start_t(55.).end_t(58.)
        
        .transform(Transformation::Translate(-4., 0., 0.))

        .rotound(-6., 60.39, 60.71, true)
        .rotound(-6., 61.03, 61.35, true)
        .rotound(-2., 61.35, 61.67, false)
        .rotound(2., 61.67, 61.99, true)

        .rotound(6., 64.06, 64.38, false)
        .rotound(6., 64.80, 65.12, false)
        .rotound(2., 65.12, 65.44, true)
        .rotound(-2., 65.44, 65.76, false)

        .transform(Transformation::Translate(0., -1., 0.)).start_t(67.33).end_t(67.61).evolution_t(Evolution::FastIn)
        .transform(Transformation::Translate(0., 1., 0.)).start_t(68.45).end_t(68.73).evolution_t(Evolution::FastIn)

        .transform(Transformation::Translate(0., -2., 0.)).start_t(71.18).end_t(71.46).evolution_t(Evolution::FastIn)
        .transform(Transformation::Translate(0., 7., 0.)).start_t(72.58).end_t(72.86).evolution_t(Evolution::FastIn)
        .transform(Transformation::Translate(2., 0., 0.)).start_t(72.58).end_t(72.86).evolution_t(Evolution::FastOut)
        .transform(Transformation::Translate(0., -5., 0.)).start_t(72.86).end_t(73.14).evolution_t(Evolution::FastOut)
        .transform(Transformation::Translate(2., 0., 0.)).start_t(72.86).end_t(73.14).evolution_t(Evolution::FastIn)
    ;

    let octahedron = Model::from_models(vec![
        polygon(3, [false, true, false])
            .transform(Transformation::RotateZ(PI/2.))
            .transform(Transformation::Translate(0., 1., 0.))
            .transform(Transformation::RotateX(0.6154))
            .transform(Transformation::RotateY(PI/4.))
            .transform(Transformation::Translate(0.0, -1.5f32.sqrt(), 0.0)),
        polygon(3, [false, true, false])
            .transform(Transformation::RotateZ(PI/2.))
            .transform(Transformation::Translate(0., 1., 0.))
            .transform(Transformation::RotateX(0.6154))
            .transform(Transformation::RotateY(3.*PI/4.))
            .transform(Transformation::Translate(0.0, -1.5f32.sqrt(), 0.0)),
        polygon(3, [false, true, false])
            .transform(Transformation::RotateZ(PI/2.))
            .transform(Transformation::Translate(0., 1., 0.))
            .transform(Transformation::RotateX(0.6154))
            .transform(Transformation::RotateY(5.*PI/4.))
            .transform(Transformation::Translate(0.0, -1.5f32.sqrt(), 0.0)),
        polygon(3, [false, true, false])
            .transform(Transformation::RotateZ(PI/2.))
            .transform(Transformation::Translate(0., 1., 0.))
            .transform(Transformation::RotateX(0.6154))
            .transform(Transformation::RotateY(-PI/4.))
            .transform(Transformation::Translate(0.0, -1.5f32.sqrt(), 0.0)),
        polygon(3, [false, true, false])
            .transform(Transformation::RotateZ(-PI/2.))
            .transform(Transformation::Translate(0., -1., 0.))
            .transform(Transformation::RotateX(0.6154))
            .transform(Transformation::RotateY(PI/4.))
            .transform(Transformation::Translate(0.0, 1.5f32.sqrt(), 0.0)),
        polygon(3, [false, true, false])
            .transform(Transformation::RotateZ(-PI/2.))
            .transform(Transformation::Translate(0., -1., 0.))
            .transform(Transformation::RotateX(0.6154))
            .transform(Transformation::RotateY(3.*PI/4.))
            .transform(Transformation::Translate(0.0, 1.5f32.sqrt(), 0.0)),
        polygon(3, [false, true, false])
            .transform(Transformation::RotateZ(-PI/2.))
            .transform(Transformation::Translate(0., -1., 0.))
            .transform(Transformation::RotateX(0.6154))
            .transform(Transformation::RotateY(5.*PI/4.))
            .transform(Transformation::Translate(0.0, 1.5f32.sqrt(), 0.0)),
        polygon(3, [false, true, false])
            .transform(Transformation::RotateZ(-PI/2.))
            .transform(Transformation::Translate(0., -1., 0.))
            .transform(Transformation::RotateX(0.6154))
            .transform(Transformation::RotateY(-PI/4.))
            .transform(Transformation::Translate(0.0, 1.5f32.sqrt(), 0.0)),
    ])

        .transform(Transformation::Scale(0.1, 0.1, 0.1))
        .transform(Transformation::Scale(10., 10., 10.)).start_t(15.5).end_t(17.)

        .transform(Transformation::RotateY(4.*PI)).start_t(16.5).end_t(21.5)
        .transform(Transformation::RotateX(-2.*PI)).start_t(16.5).end_t(21.5)
        .transform(Transformation::RotateY(4.*PI)).start_t(21.5).end_t(26.5)
        .transform(Transformation::RotateX(-2.*PI)).start_t(21.5).end_t(26.5)
        .transform(Transformation::RotateY(4.*PI)).start_t(26.5).end_t(31.5)
        .transform(Transformation::RotateX(-2.*PI)).start_t(26.5).end_t(31.5)
        .transform(Transformation::RotateY(4.*PI)).start_t(31.5).end_t(36.5)
        .transform(Transformation::RotateX(-2.*PI)).start_t(31.5).end_t(36.5)
        .transform(Transformation::RotateY(4.*PI)).start_t(36.5).end_t(43.)
        .transform(Transformation::RotateX(-2.*PI)).start_t(36.5).end_t(43.)

        .transform(Transformation::Translate(0., 3.0, 0.)).start_t(44.).end_t(55.)
        .transform(Transformation::RotateX(6.*PI)).start_t(44.).end_t(55.)
        .transform(Transformation::Translate(0., -3.0, 0.)).start_t(55.).end_t(58.)
        .transform(Transformation::RotateX(-6.*PI)).start_t(55.).end_t(58.)
        
        .rotound(2., 60.39, 60.71, true)
        .rotound(6., 60.71, 61.03, false)
        .rotound(6., 61.35, 61.67, false)
        .rotound(2., 61.67, 61.99, true)
        
        .rotound(-2., 64.06, 64.38, false)
        .rotound(-6., 64.38, 64.80, true)
        .rotound(-6., 65.12, 65.44, true)
        .rotound(-2., 65.44, 65.76, false)

        .transform(Transformation::Translate(0., 3., 0.)).start_t(67.05).end_t(68.17).evolution_t(Evolution::FastIn)
        .transform(Transformation::Translate(0., -3., 0.)).start_t(68.17).end_t(69.29).evolution_t(Evolution::FastOut)

    ;

    let dodecahedron = Model::from_models(vec![
        polygon(5, [true, true, false])
            .transform(Transformation::RotateZ(PI/2.))
            .transform(Transformation::RotateX(PI/2.))
            .transform(Transformation::Translate(0., 1.309, 0.0))
            ,
        polygon(5, [true, true, false])
            .transform(Transformation::RotateZ(PI/2.))
            .transform(Transformation::Translate(0., -(PI/5.).cos(), 0.))
            .transform(Transformation::RotateX(-0.4636))
            .transform(Transformation::Translate(0., 1.309, -(PI/5.).cos()))
            ,
        polygon(5, [true, true, false])
            .transform(Transformation::RotateZ(PI/2.))
            .transform(Transformation::Translate(0., -(PI/5.).cos(), 0.))
            .transform(Transformation::RotateX(-0.4636))
            .transform(Transformation::Translate(0., 1.309, -(PI/5.).cos()))
            .transform(Transformation::RotateY(2.*PI/5.))
            ,
        polygon(5, [true, true, false])
            .transform(Transformation::RotateZ(PI/2.))
            .transform(Transformation::Translate(0., -(PI/5.).cos(), 0.))
            .transform(Transformation::RotateX(-0.4636))
            .transform(Transformation::Translate(0., 1.309, -(PI/5.).cos()))
            .transform(Transformation::RotateY(4.*PI/5.))
            ,
        polygon(5, [true, true, false])
            .transform(Transformation::RotateZ(PI/2.))
            .transform(Transformation::Translate(0., -(PI/5.).cos(), 0.))
            .transform(Transformation::RotateX(-0.4636))
            .transform(Transformation::Translate(0., 1.309, -(PI/5.).cos()))
            .transform(Transformation::RotateY(6.*PI/5.))
            ,
        polygon(5, [true, true, false])
            .transform(Transformation::RotateZ(PI/2.))
            .transform(Transformation::Translate(0., -(PI/5.).cos(), 0.))
            .transform(Transformation::RotateX(-0.4636))
            .transform(Transformation::Translate(0., 1.309, -(PI/5.).cos()))
            .transform(Transformation::RotateY(8.*PI/5.))
            ,
        polygon(5, [true, true, false])
            .transform(Transformation::RotateZ(PI/2.))
            .transform(Transformation::RotateX(PI/2.))
            .transform(Transformation::Translate(0., 1.309, 0.0))
            .transform(Transformation::RotateX(PI))
            ,
        polygon(5, [true, true, false])
            .transform(Transformation::RotateZ(PI/2.))
            .transform(Transformation::Translate(0., -(PI/5.).cos(), 0.))
            .transform(Transformation::RotateX(-0.4636))
            .transform(Transformation::Translate(0., 1.309, -(PI/5.).cos()))
            .transform(Transformation::RotateX(PI))
            ,
        polygon(5, [true, true, false])
            .transform(Transformation::RotateZ(PI/2.))
            .transform(Transformation::Translate(0., -(PI/5.).cos(), 0.))
            .transform(Transformation::RotateX(-0.4636))
            .transform(Transformation::Translate(0., 1.309, -(PI/5.).cos()))
            .transform(Transformation::RotateY(2.*PI/5.))
            .transform(Transformation::RotateX(PI))
            ,
        polygon(5, [true, true, false])
            .transform(Transformation::RotateZ(PI/2.))
            .transform(Transformation::Translate(0., -(PI/5.).cos(), 0.))
            .transform(Transformation::RotateX(-0.4636))
            .transform(Transformation::Translate(0., 1.309, -(PI/5.).cos()))
            .transform(Transformation::RotateY(4.*PI/5.))
            .transform(Transformation::RotateX(PI))
            ,
        polygon(5, [true, true, false])
            .transform(Transformation::RotateZ(PI/2.))
            .transform(Transformation::Translate(0., -(PI/5.).cos(), 0.))
            .transform(Transformation::RotateX(-0.4636))
            .transform(Transformation::Translate(0., 1.309, -(PI/5.).cos()))
            .transform(Transformation::RotateY(6.*PI/5.))
            .transform(Transformation::RotateX(PI))
            ,
        polygon(5, [true, true, false])
            .transform(Transformation::RotateZ(PI/2.))
            .transform(Transformation::Translate(0., -(PI/5.).cos(), 0.))
            .transform(Transformation::RotateX(-0.4636))
            .transform(Transformation::Translate(0., 1.309, -(PI/5.).cos()))
            .transform(Transformation::RotateY(8.*PI/5.))
            .transform(Transformation::RotateX(PI))
            ,
    ])
        
        .transform(Transformation::Scale(0.07, 0.07, 0.07))
        .transform(Transformation::Scale(10., 10., 10.)).start_t(22.75).end_t(24.25)
        
        .transform(Transformation::RotateY(4.*PI)).start_t(23.75).end_t(28.75)
        .transform(Transformation::RotateX(-2.*PI)).start_t(23.75).end_t(28.75)
        .transform(Transformation::RotateY(4.*PI)).start_t(28.75).end_t(33.75)
        .transform(Transformation::RotateX(-2.*PI)).start_t(28.75).end_t(33.75)
        .transform(Transformation::RotateY(4.*PI)).start_t(33.75).end_t(38.75)
        .transform(Transformation::RotateX(-2.*PI)).start_t(33.75).end_t(38.75)
        .transform(Transformation::RotateY(4.*PI)).start_t(38.75).end_t(43.)
        .transform(Transformation::RotateX(-2.*PI)).start_t(38.75).end_t(43.)

        .transform(Transformation::Translate(0., 3.0, 0.)).start_t(44.).end_t(55.)
        .transform(Transformation::RotateX(4.*PI)).start_t(44.).end_t(55.)
        .transform(Transformation::Translate(0., -3.0, 0.)).start_t(55.).end_t(58.)
        .transform(Transformation::RotateX(-4.*PI)).start_t(55.).end_t(58.)

        .transform(Transformation::Translate(4.0, 0., 0.))
        
        .rotound(2., 60.39, 60.71, true)
        .rotound(-2., 60.71, 61.03, false)
        .rotound(-6., 61.03, 61.35, true)
        .rotound(-6., 61.67, 61.99, true)
        
        .rotound(-2., 64.06, 64.38, false)
        .rotound(2., 64.38, 64.80, true)
        .rotound(6., 64.80, 65.12, false)
        .rotound(6., 65.44, 65.76, false)

        .transform(Transformation::Translate(0., -1., 0.)).start_t(67.61).end_t(67.89).evolution_t(Evolution::FastIn)
        .transform(Transformation::Translate(0., 1., 0.)).start_t(68.73).end_t(69.01).evolution_t(Evolution::FastIn)

        .transform(Transformation::Translate(0., -2., 0.)).start_t(71.46).end_t(71.74).evolution_t(Evolution::FastIn)
        .transform(Transformation::Translate(0., 7., 0.)).start_t(72.30).end_t(72.58).evolution_t(Evolution::FastIn)
        .transform(Transformation::Translate(-2., 0., 0.)).start_t(72.30).end_t(72.58).evolution_t(Evolution::FastOut)
        .transform(Transformation::Translate(0., -5., 0.)).start_t(72.58).end_t(72.86).evolution_t(Evolution::FastOut)
        .transform(Transformation::Translate(-2., 0., 0.)).start_t(72.58).end_t(72.86).evolution_t(Evolution::FastIn)
    ;

    let isocahedron = Model::from_models(vec![
        polygon(3, [true, false, false])
            .transform(Transformation::RotateZ(-PI/2.))
            .transform(Transformation::RotateX(0.18))
            .transform(Transformation::Translate(0., -0.25, -1.265))
            .transform(Transformation::RotateY(PI/5.))
            ,
        polygon(3, [true, false, false])
            .transform(Transformation::RotateZ(-PI/2.))
            .transform(Transformation::RotateX(0.18))
            .transform(Transformation::Translate(0., -0.25, -1.265))
            .transform(Transformation::RotateY(3.*PI/5.))
            ,
        polygon(3, [true, false, false])
            .transform(Transformation::RotateZ(-PI/2.))
            .transform(Transformation::RotateX(0.18))
            .transform(Transformation::Translate(0., -0.25, -1.265))
            .transform(Transformation::RotateY(5.*PI/5.))
            ,
        polygon(3, [true, false, false])
            .transform(Transformation::RotateZ(-PI/2.))
            .transform(Transformation::RotateX(0.18))
            .transform(Transformation::Translate(0., -0.25, -1.265))
            .transform(Transformation::RotateY(7.*PI/5.))
            ,
        polygon(3, [true, false, false])
            .transform(Transformation::RotateZ(-PI/2.))
            .transform(Transformation::RotateX(0.18))
            .transform(Transformation::Translate(0., -0.25, -1.265))
            .transform(Transformation::RotateY(9.*PI/5.))
            ,

        polygon(3, [true, false, false])
            .transform(Transformation::RotateZ(-PI/2.))
            .transform(Transformation::Translate(0., 0., -1.308))
            .transform(Transformation::RotateX(-0.918))
            ,
        polygon(3, [true, false, false])
            .transform(Transformation::RotateZ(-PI/2.))
            .transform(Transformation::Translate(0., 0., -1.308))
            .transform(Transformation::RotateX(-0.918))
            .transform(Transformation::RotateY(2.*PI/5.))
            ,
        polygon(3, [true, false, false])
            .transform(Transformation::RotateZ(-PI/2.))
            .transform(Transformation::Translate(0., 0., -1.308))
            .transform(Transformation::RotateX(-0.918))
            .transform(Transformation::RotateY(4.*PI/5.))
            ,
        polygon(3, [true, false, false])
            .transform(Transformation::RotateZ(-PI/2.))
            .transform(Transformation::Translate(0., 0., -1.308))
            .transform(Transformation::RotateX(-0.918))
            .transform(Transformation::RotateY(6.*PI/5.))
            ,
        polygon(3, [true, false, false])
            .transform(Transformation::RotateZ(-PI/2.))
            .transform(Transformation::Translate(0., 0., -1.308))
            .transform(Transformation::RotateX(-0.918))
            .transform(Transformation::RotateY(8.*PI/5.))
            ,

        polygon(3, [true, false, false])
            .transform(Transformation::RotateZ(-PI/2.))
            .transform(Transformation::RotateX(0.18))
            .transform(Transformation::Translate(0., -0.25, -1.265))
            .transform(Transformation::RotateY(PI/5.))
            .transform(Transformation::RotateX(PI))
            ,
        polygon(3, [true, false, false])
            .transform(Transformation::RotateZ(-PI/2.))
            .transform(Transformation::RotateX(0.18))
            .transform(Transformation::Translate(0., -0.25, -1.265))
            .transform(Transformation::RotateY(3.*PI/5.))
            .transform(Transformation::RotateX(PI))
            ,
        polygon(3, [true, false, false])
            .transform(Transformation::RotateZ(-PI/2.))
            .transform(Transformation::RotateX(0.18))
            .transform(Transformation::Translate(0., -0.25, -1.265))
            .transform(Transformation::RotateY(5.*PI/5.))
            .transform(Transformation::RotateX(PI))
            ,
        polygon(3, [true, false, false])
            .transform(Transformation::RotateZ(-PI/2.))
            .transform(Transformation::RotateX(0.18))
            .transform(Transformation::Translate(0., -0.25, -1.265))
            .transform(Transformation::RotateY(7.*PI/5.))
            .transform(Transformation::RotateX(PI))
            ,
        polygon(3, [true, false, false])
            .transform(Transformation::RotateZ(-PI/2.))
            .transform(Transformation::RotateX(0.18))
            .transform(Transformation::Translate(0., -0.25, -1.265))
            .transform(Transformation::RotateY(9.*PI/5.))
            .transform(Transformation::RotateX(PI))
            ,

        polygon(3, [true, false, false])
            .transform(Transformation::RotateZ(-PI/2.))
            .transform(Transformation::Translate(0., 0., -1.308))
            .transform(Transformation::RotateX(-0.918))
            .transform(Transformation::RotateX(PI))
            ,
        polygon(3, [true, false, false])
            .transform(Transformation::RotateZ(-PI/2.))
            .transform(Transformation::Translate(0., 0., -1.308))
            .transform(Transformation::RotateX(-0.918))
            .transform(Transformation::RotateY(2.*PI/5.))
            .transform(Transformation::RotateX(PI))
            ,
        polygon(3, [true, false, false])
            .transform(Transformation::RotateZ(-PI/2.))
            .transform(Transformation::Translate(0., 0., -1.308))
            .transform(Transformation::RotateX(-0.918))
            .transform(Transformation::RotateY(4.*PI/5.))
            .transform(Transformation::RotateX(PI))
            ,
        polygon(3, [true, false, false])
            .transform(Transformation::RotateZ(-PI/2.))
            .transform(Transformation::Translate(0., 0., -1.308))
            .transform(Transformation::RotateX(-0.918))
            .transform(Transformation::RotateY(6.*PI/5.))
            .transform(Transformation::RotateX(PI))
            ,
        polygon(3, [true, false, false])
            .transform(Transformation::RotateZ(-PI/2.))
            .transform(Transformation::Translate(0., 0., -1.308))
            .transform(Transformation::RotateX(-0.918))
            .transform(Transformation::RotateY(8.*PI/5.))
            .transform(Transformation::RotateX(PI))
            ,
    ])

        .transform(Transformation::Scale(0.07, 0.07, 0.07))
        .transform(Transformation::Scale(10., 10., 10.)).start_t(30.).end_t(31.5)
        .transform(Transformation::RotateY(PI/5.)).start_t(71.74).end_t(72.30).evolution_t(Evolution::FastIn)
        
        .transform(Transformation::RotateY(4.*PI)).start_t(31.).end_t(36.)
        .transform(Transformation::RotateX(-2.*PI)).start_t(31.).end_t(36.)
        .transform(Transformation::RotateY(4.*PI)).start_t(36.).end_t(43.)
        .transform(Transformation::RotateX(-2.*PI)).start_t(36.).end_t(43.)

        .transform(Transformation::Translate(0., 3.0, 0.)).start_t(44.).end_t(55.)
        .transform(Transformation::RotateX(2.*PI)).start_t(44.).end_t(55.)
        .transform(Transformation::Translate(0., -3.0, 0.)).start_t(55.).end_t(58.)
        .transform(Transformation::RotateX(-2.*PI)).start_t(55.).end_t(58.)

        .transform(Transformation::Translate(8., 0., 0.))
        
        .rotound(6., 60.71, 61.03, false)
        .rotound(2., 61.03, 61.35, true)
        .rotound(-2., 61.35, 61.67, false)
        .rotound(-6., 61.67, 61.99, true)
        
        .rotound(-6., 64.38, 64.80, true)
        .rotound(-2., 64.80, 65.12, false)
        .rotound(2., 65.12, 65.44, true)
        .rotound(6., 65.44, 65.76, false)

        .transform(Transformation::Translate(0., -1., 0.)).start_t(67.89).end_t(68.17).evolution_t(Evolution::FastIn)
        .transform(Transformation::Translate(0., 1., 0.)).start_t(69.01).end_t(69.29).evolution_t(Evolution::FastIn)

        .transform(Transformation::Translate(0., 1., 0.)).start_t(70.62).end_t(70.90).evolution_t(Evolution::FastIn)
        .transform(Transformation::Translate(0., -5., 0.)).start_t(71.74).end_t(72.02).evolution_t(Evolution::FastIn)
        .transform(Transformation::Translate(-2., 0., 0.)).start_t(71.74).end_t(72.02).evolution_t(Evolution::FastOut)
        .transform(Transformation::Translate(0., 2., 0.)).start_t(72.02).end_t(72.30).evolution_t(Evolution::FastOut)
        .transform(Transformation::Translate(-2., 0., 0.)).start_t(72.02).end_t(72.30).evolution_t(Evolution::FastIn)
        .transform(Transformation::Translate(0., 7., 0.)).start_t(72.30).end_t(72.58).evolution_t(Evolution::FastIn)
        .transform(Transformation::Translate(-2., 0., 0.)).start_t(72.30).end_t(72.58).evolution_t(Evolution::FastOut)
        .transform(Transformation::Translate(0., -5., 0.)).start_t(72.58).end_t(72.86).evolution_t(Evolution::FastOut)
        .transform(Transformation::Translate(-2., 0., 0.)).start_t(72.58).end_t(72.86).evolution_t(Evolution::FastIn)
    ;


        
        // .rotound(6., 64.38, 64.80, false)
        // .rotound(2., 64.80, 65.12, true)
        // .rotound(-2., 65.12, 65.44, false)
        // .rotound(-6., 65.44, 65.76, true)

        // 
        // .rotound(2., 64.06, 64.38, true)
        // .rotound(-2., 64.38, 64.80, false)
        // .rotound(-6., 64.80, 65.12, true)
        // .rotound(-6., 65.44, 65.76, true)

        // 
        // .rotound(2., 64.06, 64.38, true)
        // .rotound(6., 64.38, 64.80, false)
        // .rotound(6., 65.12, 65.44, false)
        // .rotound(2., 65.44, 65.76, true)


        // .rotound(-6., 64.06, 64.38, true)
        // .rotound(-6., 64.80, 65.12, true)
        // .rotound(-2., 65.12, 65.44, false)
        // .rotound(2., 65.44, 65.76, true)

        // .rotound(-6., 64.06, 64.38, true)
        // .rotound(-2., 64.38, 64.80, false)
        // .rotound(2., 64.80, 65.12, true)
        // .rotound(6., 65.12, 65.44, false)

    Input {
        meta: MetaInput {
            bg: [0.1, 0., 0.1, 1.],
            start: -10.,
            end: 90.,
        },
        m: vec![
            Model::from_models(
                vec![
                    tetrahedron,
                    cube,
                    octahedron,
                    dodecahedron,
                    isocahedron,
                ]
            )
                .transform(Transformation::RotateY(PI)).start_t(59.75).end_t(60.39).evolution_t(Evolution::Linear)
                .transform(Transformation::RotateY(-PI)).start_t(63.42).end_t(64.06).evolution_t(Evolution::Linear)
                .transform(Transformation::RotateY(-PI)).start_t(74.13).end_t(75.57).evolution_t(Evolution::FastOut)
                .transform(Transformation::Scale(2., 2., 2.)).start_t(74.13).end_t(74.85).evolution_t(Evolution::FastIn)
                .transform(Transformation::Scale(0., 0., 0.)).start_t(74.85).end_t(75.1).evolution_t(Evolution::FastIn)
        ],
        v: View::new()
            .transform(Transformation::Lookat(-8., 0., 6., -8., 0., 0., 0., 1., 0.))

            .transform(Transformation::Lookat(-6., 0., 5., -8., 0., 0., 0., 1., 0.)).start_t(7.).end_t(8.).evolution_t(Evolution::Linear)
            .transform(Transformation::Lookat(-6., 0., 5., -4., 0., 0., 0., 1., 0.)).start_t(7.).end_t(8.).evolution_t(Evolution::Linear)
            .transform(Transformation::Lookat(-4., 0., 6., -4., 0., 0., 0., 1., 0.)).start_t(7.).end_t(8.).evolution_t(Evolution::Linear)
            .transform(Transformation::Lookat(-2., 0., 5., -4., 0., 0., 0., 1., 0.)).start_t(14.25).end_t(15.25).evolution_t(Evolution::Linear)
            .transform(Transformation::Lookat(-2., 0., 5., 0., 0., 0., 0., 1., 0.)).start_t(14.25).end_t(15.25).evolution_t(Evolution::Linear)
            .transform(Transformation::Lookat(0., 0., 6., 0., 0., 0., 0., 1., 0.)).start_t(14.25).end_t(15.25).evolution_t(Evolution::Linear)
            .transform(Transformation::Lookat(2., 0., 5., 0., 0., 0., 0., 1., 0.)).start_t(21.5).end_t(22.5).evolution_t(Evolution::Linear)
            .transform(Transformation::Lookat(2., 0., 5., 4., 0., 0., 0., 1., 0.)).start_t(21.5).end_t(22.5).evolution_t(Evolution::Linear)
            .transform(Transformation::Lookat(4., 0., 6., 4., 0., 0., 0., 1., 0.)).start_t(21.5).end_t(22.5).evolution_t(Evolution::Linear)
            .transform(Transformation::Lookat(6., 0., 5., 4., 0., 0., 0., 1., 0.)).start_t(28.75).end_t(29.75).evolution_t(Evolution::Linear)
            .transform(Transformation::Lookat(6., 0., 5., 8., 0., 0., 0., 1., 0.)).start_t(28.75).end_t(29.75).evolution_t(Evolution::Linear)
            .transform(Transformation::Lookat(8., 0., 6., 8., 0., 0., 0., 1., 0.)).start_t(28.75).end_t(29.75).evolution_t(Evolution::Linear)
            .transform(Transformation::Lookat(10., 0., 5., 8., 0., 0., 0., 1., 0.)).start_t(36.).end_t(37.).evolution_t(Evolution::Linear)
            .transform(Transformation::Lookat(10., 0., 5., 12., 0., 0., 0., 1., 0.)).start_t(36.).end_t(37.).evolution_t(Evolution::Linear)
            .transform(Transformation::Lookat(0., -5., 10., 0., 0., 0., 0., 1., 0.)).start_t(36.).end_t(37.).evolution_t(Evolution::FastIn)

            .transform(Transformation::Lookat(0., -10., 15., 0., 0., 0., 0., 1., 0.)).start_t(44.).end_t(45.).evolution_t(Evolution::FastIn)
            .transform(Transformation::Lookat(-12., 3., 0., -3., 0., 0., 0., 1., 0.)).start_t(44.).end_t(46.).evolution_t(Evolution::FastMiddle)
            .transform(Transformation::Lookat(0., 0., -10., 0., 0., 0., 0., 1., 0.)).start_t(44.).end_t(48.).evolution_t(Evolution::FastMiddle)
            .transform(Transformation::Lookat(15., 0., -5., 3., 0., 0., 0., 1., 0.)).start_t(44.).end_t(50.).evolution_t(Evolution::FastMiddle)
            .transform(Transformation::Lookat(15., 0., 0., 3., 3., 0., 0., 1., 0.)).start_t(44.).end_t(55.).evolution_t(Evolution::FastMiddle)
            .transform(Transformation::Lookat(18., 0., 0., 0., 0., 0., 0., 1., 0.)).start_t(55.).end_t(56.).evolution_t(Evolution::FastIn)

            .transform(Transformation::Lookat(0., -5., 17., 0., 0., 0., 0., 1., 0.)).start_t(57.58).end_t(59.75).evolution_t(Evolution::FastMiddle)
            .transform(Transformation::Lookat(0., -2.5, 8.5, 0., 0., 0., 0., 1., 0.)).start_t(71.74).end_t(73.14).evolution_t(Evolution::FastOut)
            ,
        p: Projection::new().transform(Transformation::Perspective(-1., 1., -1., 1., 2., 100.)),
    }
}

fn polygon(n: u16, color_channels: [bool; 3]) -> Model {
    Model::from_vertices(
        (0..n).flat_map(|i| vec![
            Vertex::new().pos(((i as f32 / n as f32) * 2. * PI).cos(), ((i as f32 / n as f32) * 2. * PI).sin(), 0.),
            Vertex::new().pos((((i as f32 + 1.) / n as f32) * 2. * PI).cos(), (((i as f32 + 1.) / n as f32) * 2. * PI).sin(), 0.),
            Vertex::new().pos(0., 0., 0.),
        ]).collect()
    )
        .rgb(if color_channels[0] {(fastrand::f32()+1.)/2.} else { 0. }, if color_channels[1] {(fastrand::f32()+1.)/2.} else { 0. }, if color_channels[2] {(fastrand::f32()+1.)/2.} else { 0. })
}

trait Rot<T> {
    fn rotound(self: Self, x_offset: f32, start: f32, end: f32, cw: bool) -> Self;
}

impl Rot<Model> for Model {
    // Rotates linearly by Pi around a y axis with an X offset from START to END CLOCKWISE
    fn rotound(self, x_offset: f32, start: f32, end: f32, cw: bool) -> Self {
        self
            .transform(Transformation::Translate(-x_offset, 0., 0.)).start_t(start).end_t(start)
            .transform(Transformation::RotateY(if cw { PI } else { -PI })).start_t(start).end_t(end).evolution_t(Evolution::Linear)
            .transform(Transformation::Translate(x_offset, 0., 0.)).start_t(start).end_t(start)
    }
}