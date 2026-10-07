//! Minimal DXF fixtures written by hand, to pin down parser behaviour that
//! round-trip tests cannot catch (they would happily accept a symmetric bug).

use cad_core::Vec2;
use cad_doc::{Document, EntityKind};
use cad_geom::curve::{Arc, Circle, Line};

fn doc() -> Document {
    Document::new()
}

#[test]
fn circle_from_hand_written_pairs() {
    let text = "\
0
SECTION
2
ENTITIES
0
CIRCLE
8
0
10
1.5
20
2.5
40
7.25
0
ENDSEC
0
EOF
";
    let mut d = doc();
    let rep = cad_io::read(&mut d, text).unwrap();
    assert_eq!(rep.entities, 1);
    let e = d.entities.iter().next().unwrap();
    match &e.entity {
        EntityKind::Circle(c) => {
            assert!(
                c.center.distance(Vec2::new(1.5, 2.5)) < 1e-6,
                "{:?}",
                c.center
            );
            assert!((c.radius - 7.25).abs() < 1e-6);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn arc_sweep_direction_is_preserved() {
    // A 270 degree arc drawn clockwise: the sweep must be negative.
    let text = "\
0
SECTION
2
ENTITIES
0
ARC
8
0
10
0.0
20
0.0
40
10.0
50
0.0
51
-90.0
0
ENDSEC
0
EOF
";
    let mut d = doc();
    cad_io::read(&mut d, text).unwrap();
    let e = d.entities.iter().next().unwrap();
    match &e.entity {
        EntityKind::Arc(a) => {
            assert!(a.sweep < 0.0, "sweep should be clockwise: {}", a.sweep);
            assert!((a.sweep + std::f32::consts::FRAC_PI_2).abs() < 1e-5);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn lwpolyline_vertices_and_bulges() {
    // Three vertices, a bulge on the first segment only.
    let text = "\
0
SECTION
2
ENTITIES
0
LWPOLYLINE
8
0
90
3
70
1
10
0.0
20
0.0
42
0.5
10
10.0
20
0.0
10
10.0
20
10.0
0
ENDSEC
0
EOF
";
    let mut d = doc();
    let rep = cad_io::read(&mut d, text).unwrap();
    assert_eq!(rep.entities, 1);
    let e = d.entities.iter().next().unwrap();
    match &e.entity {
        EntityKind::Polyline(p) => {
            assert_eq!(p.vertices.len(), 3);
            assert!(p.closed);
            assert!(p.vertices[1].distance(Vec2::new(10.0, 0.0)) < 1e-5);
            assert!(p.vertices[2].distance(Vec2::new(10.0, 10.0)) < 1e-5);
            assert!((p.bulge_at(0).0 - 0.5).abs() < 1e-6);
            assert!(!p.bulge_at(1).is_arc());
            assert!(!p.bulge_at(2).is_arc());
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn spline_control_points_become_a_curve() {
    let text = "\
0
SECTION
2
ENTITIES
0
SPLINE
8
0
70
8
10
0.0
20
0.0
30
0.0
10
5.0
20
10.0
30
0.0
10
10.0
20
0.0
30
0.0
0
ENDSEC
0
EOF
";
    let mut d = doc();
    let rep = cad_io::read(&mut d, text).unwrap();
    assert_eq!(rep.entities, 1);
    let e = d.entities.iter().next().unwrap();
    match &e.entity {
        EntityKind::Spline(s) => {
            assert_eq!(s.segments.len(), 2);
            assert!(s.length() > 15.0, "length={}", s.length());
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn insert_references_a_known_block() {
    let text = "\
0
SECTION
2
BLOCKS
0
BLOCK
2
DOOR
70
0
10
0.0
20
0.0
30
0.0
0
LINE
8
0
10
0.0
20
0.0
11
1.0
21
0.0
0
ENDBLK
0
ENDSEC
0
SECTION
2
ENTITIES
0
INSERT
8
0
2
DOOR
10
5.0
20
6.0
30
0.0
41
2.0
42
2.0
43
2.0
0
ENDSEC
0
EOF
";
    let mut d = doc();
    let rep = cad_io::read(&mut d, text).unwrap();
    assert_eq!(rep.blocks, 1, "{rep:?}");
    assert_eq!(rep.entities, 1);

    let block = d.blocks.by_name("DOOR").expect("block not registered");
    assert_eq!(d.blocks.by_id(block).unwrap().entities.len(), 1);

    let e = d.entities.iter().next().unwrap();
    match &e.entity {
        EntityKind::Insert(i) => {
            assert_eq!(i.block, block);
            assert!((i.scale.x - 2.0).abs() < 1e-6);
            assert!(i.position.distance(cad_core::Vec3::new(5.0, 6.0, 0.0)) < 1e-5);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn degenerate_entities_are_dropped_not_panicked() {
    // A zero-radius arc and a one-vertex polyline are both invalid.
    let text = "\
0
SECTION
2
ENTITIES
0
ARC
8
0
10
0.0
20
0.0
40
0.0
50
0.0
51
90.0
0
LWPOLYLINE
8
0
90
1
10
5.0
20
5.0
0
LINE
8
0
10
0.0
20
0.0
11
10.0
21
10.0
0
ENDSEC
0
EOF
";
    let mut d = doc();
    let rep = cad_io::read(&mut d, text).unwrap();
    assert_eq!(rep.entities, 1, "{rep:?}");
    let mut lines = 0;
    for e in d.entities.iter() {
        if let EntityKind::Line(l) = &e.entity {
            lines += 1;
            assert!(l.p1.distance(Vec2::new(10.0, 10.0)) < 1e-5);
        }
    }
    assert_eq!(lines, 1);
}

#[test]
fn section_order_does_not_matter() {
    // ENTITIES before TABLES in the file, which some exporters emit.
    let text = "\
0
SECTION
2
ENTITIES
0
CIRCLE
8
Custom
10
0.0
20
0.0
40
1.0
0
ENDSEC
0
SECTION
2
TABLES
0
TABLE
2
LAYER
70
1
0
LAYER
2
Custom
70
0
62
5
0
ENDTAB
0
ENDSEC
0
EOF
";
    let mut d = doc();
    let rep = cad_io::read(&mut d, text).unwrap();
    assert_eq!(rep.entities, 1);
    let custom = d.layers.by_name("CUSTOM").expect("layer not registered");
    assert_eq!(d.entities.iter().next().unwrap().layer(), custom);
    assert_eq!(d.layers.by_id(custom).unwrap().color, 5);
}

#[test]
fn line_endpoints_use_the_11_21_codes() {
    // A naive parser that only reads 10/20 gets this wrong; ours must not.
    let text = "\
0
SECTION
2
ENTITIES
0
LINE
8
0
10
1.0
20
2.0
30
0.0
11
100.0
21
200.0
31
0.0
0
ENDSEC
0
EOF
";
    let mut d = doc();
    cad_io::read(&mut d, text).unwrap();
    let e = d.entities.iter().next().unwrap();
    match &e.entity {
        EntityKind::Line(l) => {
            assert!(l.p0.distance(Vec2::new(1.0, 2.0)) < 1e-5, "p0={:?}", l.p0);
            assert!(
                l.p1.distance(Vec2::new(100.0, 200.0)) < 1e-5,
                "p1={:?}",
                l.p1
            );
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn frozen_layer_flag_is_decoded() {
    let text = "\
0
SECTION
2
TABLES
0
TABLE
2
LAYER
70
1
0
LAYER
2
Frozen
70
0
62
7
290
1
0
ENDTAB
0
ENDSEC
0
EOF
";
    let mut d = doc();
    cad_io::read(&mut d, text).unwrap();
    let id = d.layers.by_name("FROZEN").unwrap();
    assert!(d.layers.by_id(id).unwrap().frozen);
}
