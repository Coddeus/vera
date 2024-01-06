use vera::{Input, MetaInput, Model, Vertex, View, Projection};

use crate::Vera;

fn receive_input() {
    let v = Vera::init(Input {
        meta: MetaInput {
            bg: [0.0, 0.1, 0.0, 1.0],
            start: 0.1,
            end: 1.1,
        },
        m: vec![
            Model::from_vm(
                vec![
                    Vertex::new(),
                    Vertex::new(),
                    Vertex::new(),
                ],
                vec![
                    Model::from_vertices(vec![
                        Vertex::new(),
                        Vertex::new(),
                        Vertex::new(),
                    ]),
                    Model::from_vertices(vec![
                        Vertex::new(),
                        Vertex::new(),
                        Vertex::new(),
                    ]),
                ],
            )
        ],
        v: View::new(),
        p: Projection::new(),
    });

    // Meta
    assert!(
        v.vk.background_color == [0.0, 0.1, 0.0, 1.0],
    );
    assert!(
        v.vk.start_time == 0.1,
    );
    assert!(
        v.vk.end_time == 1.1,
    );

    // Buffers
}