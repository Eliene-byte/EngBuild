//! Round-trip through a real DXF file on disk, plus parser behaviour on
//! hand-written fixtures.

use cad_core::Vec2;
use cad_doc::{Document, Entity, EntityKind};
use cad_geom::curve::{Circle, Line};

#[test]
fn reads_a_hand_written_dxf() {
    let mut doc = Document::new();
    let rep = cad_io::read(&mut doc, include_str!("sample.dxf")).expect("parse failed");
    assert_eq!(rep.entities, 1, "{rep:?}");
    assert!(doc.layers.by_name("PAREDE").is_some());
    let e = doc.entities.iter().next().unwrap();
    assert_eq!(e.kind(), "CIRCLE");
    match &e.entity {
        EntityKind::Circle(c) => {
            assert!(
                c.center.distance(Vec2::new(3.0, 4.0)) < 1e-3,
                "{:?}",
                c.center
            );
            assert!((c.radius - 2.5).abs() < 1e-3);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn entity_layer_is_resolved_by_name() {
    let mut doc = Document::new();
    cad_io::read(&mut doc, include_str!("sample.dxf")).unwrap();
    let layer = doc.layers.by_name("PAREDE").unwrap();
    let e = doc.entities.iter().next().unwrap();
    assert_eq!(e.layer(), layer);
}

#[test]
fn file_on_disk_round_trips() {
    let dir = std::env::temp_dir().join("cadkit-dxf-test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("round.dxf");

    let mut d = Document::new();
    let base = d.layers.ensure_default();
    let walls = d.layers.insert(cad_doc::Layer::new("Parede"));
    d.add(Entity::line(Line::new(Vec2::ZERO, Vec2::new(100.0, 0.0))).with_layer(base));
    d.add(Entity::circle(Circle::new(Vec2::new(50.0, 50.0), 25.0)).with_layer(walls));
    cad_io::dxf::save(&d, &path).unwrap();

    let mut back = Document::new();
    let rep = cad_io::import(&mut back, &path).expect("import failed");
    assert_eq!(rep.entities, 2);
    assert_eq!(back.entities.len(), 2);

    // The written file must be a legal pair stream.
    let text = std::fs::read_to_string(&path).unwrap();
    assert_eq!(text.lines().count() % 2, 0);

    std::fs::remove_file(&path).ok();
}

#[test]
fn dxf_and_native_agree_on_entity_count() {
    let mut d = Document::new();
    let layer = d.layers.ensure_default();
    for i in 0..50 {
        d.add(
            Entity::circle(Circle::new(Vec2::new(i as f32 * 3.0, i as f32 * 2.0), 1.0))
                .with_layer(layer),
        );
    }
    let dxf = cad_io::dxf::write(&d);
    let mut from_dxf = Document::new();
    let a = cad_io::read(&mut from_dxf, &dxf).unwrap();

    let native = cad_io::native::encode(&d);
    let mut from_native = Document::new();
    cad_io::native::decode(&native, &mut from_native).unwrap();

    assert_eq!(a.entities, 50);
    assert_eq!(from_native.entities.len(), 50);
}
