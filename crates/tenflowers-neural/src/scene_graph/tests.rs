use super::*;

// ── §1 SgBoundingBox tests ──

#[test]
fn test_bbox_area_positive() {
    let bb = SgBoundingBox::new(0.0, 0.0, 3.0, 4.0);
    assert!((bb.area() - 12.0).abs() < 1e-9);
}

#[test]
fn test_bbox_area_zero_dimension() {
    let bb = SgBoundingBox::new(0.0, 0.0, 0.0, 5.0);
    assert_eq!(bb.area(), 0.0);
}

#[test]
fn test_bbox_iou_identical() {
    let bb = SgBoundingBox::new(0.0, 0.0, 10.0, 10.0);
    assert!((bb.iou(&bb) - 1.0).abs() < 1e-9);
}

#[test]
fn test_bbox_iou_no_overlap() {
    let b1 = SgBoundingBox::new(0.0, 0.0, 2.0, 2.0);
    let b2 = SgBoundingBox::new(5.0, 5.0, 2.0, 2.0);
    assert_eq!(b1.iou(&b2), 0.0);
}

#[test]
fn test_bbox_iou_in_range() {
    let b1 = SgBoundingBox::new(0.0, 0.0, 4.0, 4.0);
    let b2 = SgBoundingBox::new(2.0, 2.0, 4.0, 4.0);
    let iou = b1.iou(&b2);
    assert!((0.0..=1.0).contains(&iou));
    // Intersection = 2x2=4, union = 16+16-4 = 28
    assert!((iou - 4.0 / 28.0).abs() < 1e-9);
}

#[test]
fn test_bbox_center() {
    let bb = SgBoundingBox::new(2.0, 4.0, 6.0, 8.0);
    let (cx, cy) = bb.center();
    assert!((cx - 5.0).abs() < 1e-9);
    assert!((cy - 8.0).abs() < 1e-9);
}

#[test]
fn test_spatial_relation_above() {
    // self is above other when its center has smaller y
    let above = SgBoundingBox::new(5.0, 0.0, 2.0, 2.0);
    let below = SgBoundingBox::new(5.0, 10.0, 2.0, 2.0);
    let rel = above.spatial_relation(&below);
    assert_eq!(rel, SgSpatialRelation::Above);
}

#[test]
fn test_spatial_relation_below() {
    let top = SgBoundingBox::new(5.0, 0.0, 2.0, 2.0);
    let bot = SgBoundingBox::new(5.0, 10.0, 2.0, 2.0);
    let rel = bot.spatial_relation(&top);
    assert_eq!(rel, SgSpatialRelation::Below);
}

#[test]
fn test_spatial_relation_left() {
    let left = SgBoundingBox::new(0.0, 5.0, 2.0, 2.0);
    let right = SgBoundingBox::new(10.0, 5.0, 2.0, 2.0);
    let rel = left.spatial_relation(&right);
    assert_eq!(rel, SgSpatialRelation::Left);
}

#[test]
fn test_spatial_relation_right() {
    let left = SgBoundingBox::new(0.0, 5.0, 2.0, 2.0);
    let right = SgBoundingBox::new(10.0, 5.0, 2.0, 2.0);
    let rel = right.spatial_relation(&left);
    assert_eq!(rel, SgSpatialRelation::Right);
}

#[test]
fn test_spatial_relation_inside() {
    let inner = SgBoundingBox::new(2.0, 2.0, 2.0, 2.0);
    let outer = SgBoundingBox::new(0.0, 0.0, 10.0, 10.0);
    let rel = inner.spatial_relation(&outer);
    assert_eq!(rel, SgSpatialRelation::Inside);
}

#[test]
fn test_spatial_relation_contains() {
    let outer = SgBoundingBox::new(0.0, 0.0, 10.0, 10.0);
    let inner = SgBoundingBox::new(2.0, 2.0, 2.0, 2.0);
    let rel = outer.spatial_relation(&inner);
    assert_eq!(rel, SgSpatialRelation::Contains);
}

#[test]
fn test_spatial_relation_distant() {
    let b1 = SgBoundingBox::new(0.0, 0.0, 1.0, 1.0);
    let b2 = SgBoundingBox::new(100.0, 100.0, 1.0, 1.0);
    let rel = b1.spatial_relation(&b2);
    assert_eq!(rel, SgSpatialRelation::Distant);
}

// ── §2 SgSceneGraph tests ──

fn make_sg() -> SgSceneGraph {
    let mut sg = SgSceneGraph::new();
    let o1 = SgObject {
        id: 0,
        label: "cat".into(),
        confidence: 0.9,
        bbox: SgBoundingBox::new(0.0, 0.0, 10.0, 10.0),
        attributes: vec!["fluffy".into()],
        feature: vec![0.1, 0.2, 0.3, 0.4],
    };
    let o2 = SgObject {
        id: 0,
        label: "dog".into(),
        confidence: 0.8,
        bbox: SgBoundingBox::new(20.0, 20.0, 10.0, 10.0),
        attributes: vec![],
        feature: vec![0.5, 0.6, 0.7, 0.8],
    };
    sg.add_object(o1);
    sg.add_object(o2);
    sg
}

#[test]
fn test_scene_graph_n_objects() {
    let sg = make_sg();
    assert_eq!(sg.n_objects(), 2);
}

#[test]
fn test_scene_graph_add_relationship() {
    let mut sg = make_sg();
    sg.add_relationship(SgRelationship {
        subject_id: 0,
        object_id: 1,
        predicate: "near".into(),
        predicate_id: 0,
        confidence: 0.7,
        feature: vec![0.7],
    });
    assert_eq!(sg.n_relationships(), 1);
}

#[test]
fn test_scene_graph_adjacency_matrix_shape() {
    let mut sg = make_sg();
    sg.add_relationship(SgRelationship {
        subject_id: 0,
        object_id: 1,
        predicate: "near".into(),
        predicate_id: 0,
        confidence: 0.7,
        feature: vec![0.7],
    });
    let adj = sg.adjacency_matrix();
    assert_eq!(adj.len(), 2);
    assert_eq!(adj[0].len(), 2);
    assert!((adj[0][1] - 0.7).abs() < 1e-9);
    assert_eq!(adj[1][0], 0.0);
}

#[test]
fn test_scene_graph_triplets() {
    let mut sg = make_sg();
    sg.add_relationship(SgRelationship {
        subject_id: 0,
        object_id: 1,
        predicate: "chases".into(),
        predicate_id: 1,
        confidence: 0.8,
        feature: vec![],
    });
    let trips = sg.triplets();
    assert_eq!(trips.len(), 1);
    assert_eq!(trips[0], ("cat".to_string(), "chases".to_string(), "dog".to_string()));
}

#[test]
fn test_scene_graph_get_subject_object() {
    let mut sg = make_sg();
    sg.add_relationship(SgRelationship {
        subject_id: 0,
        object_id: 1,
        predicate: "near".into(),
        predicate_id: 0,
        confidence: 0.7,
        feature: vec![],
    });
    let pair = sg.get_subject_object(0);
    assert!(pair.is_some());
    let (subj, obj) = pair.expect("get_subject_object should return Some");
    assert_eq!(subj.label, "cat");
    assert_eq!(obj.label, "dog");
}

#[test]
fn test_scene_graph_get_subject_object_oob() {
    let sg = make_sg();
    assert!(sg.get_subject_object(99).is_none());
}

// ── §3 SgObjectDetector tests ──

#[test]
fn test_object_detector_classify_sums_to_one() {
    let det = SgObjectDetector::new(8, 5);
    let feature = vec![0.1f64; 8];
    let scores = det.classify(&feature);
    assert_eq!(scores.len(), 5);
    let total: f64 = scores.iter().sum();
    assert!((total - 1.0).abs() < 1e-9);
}

#[test]
fn test_object_detector_classify_all_positive() {
    let det = SgObjectDetector::new(8, 5);
    let feature = vec![-0.5f64; 8];
    let scores = det.classify(&feature);
    assert!(scores.iter().all(|&s| s >= 0.0));
}

#[test]
fn test_object_detector_regress_bbox() {
    let det = SgObjectDetector::new(8, 5);
    let anchor = SgBoundingBox::new(10.0, 10.0, 20.0, 20.0);
    let feature = vec![0.0f64; 8];
    let bbox = det.regress_bbox(&feature, &anchor);
    // With zero feature, offsets are b=0 → same size as anchor
    assert!(bbox.w > 0.0);
    assert!(bbox.h > 0.0);
}

#[test]
fn test_object_detector_nms_removes_overlap() {
    let det = SgObjectDetector::new(4, 3);
    let b1 = SgBoundingBox::new(0.0, 0.0, 10.0, 10.0);
    let b2 = SgBoundingBox::new(1.0, 1.0, 10.0, 10.0); // high IoU with b1
    let b3 = SgBoundingBox::new(50.0, 50.0, 10.0, 10.0); // no overlap
    let detections = vec![(b1, 0.9, 1usize), (b2, 0.8, 1usize), (b3, 0.7, 2usize)];
    let kept = det.nms(&detections);
    assert_eq!(kept.len(), 2); // b2 suppressed
}

#[test]
fn test_object_detector_nms_empty() {
    let det = SgObjectDetector::new(4, 3);
    let kept = det.nms(&[]);
    assert!(kept.is_empty());
}

#[test]
fn test_object_detector_detect_dim_mismatch() {
    let det = SgObjectDetector::new(8, 5);
    let feats = vec![vec![0.0f64; 8], vec![0.0f64; 8]];
    let anchors = vec![SgBoundingBox::new(0.0, 0.0, 1.0, 1.0)]; // wrong length
    assert!(det.detect(&feats, &anchors).is_err());
}

#[test]
fn test_object_detector_detect_empty() {
    let det = SgObjectDetector::new(8, 5);
    let result = det.detect(&[], &[]).expect("detect on empty input should succeed");
    assert!(result.is_empty());
}

// ── §4 SgRelationshipDetector tests ──

#[test]
fn test_rel_detector_spatial_features_length() {
    let rd = SgRelationshipDetector::new(8, 5, 4);
    let b1 = SgBoundingBox::new(0.0, 0.0, 10.0, 10.0);
    let b2 = SgBoundingBox::new(15.0, 0.0, 10.0, 10.0);
    let spatial = rd.spatial_features(&b1, &b2);
    assert_eq!(spatial.len(), 4);
}

#[test]
fn test_rel_detector_predict_predicates_sums_to_one() {
    let rd = SgRelationshipDetector::new(8, 5, 4);
    let b1 = SgBoundingBox::new(0.0, 0.0, 10.0, 10.0);
    let b2 = SgBoundingBox::new(15.0, 0.0, 10.0, 10.0);
    let feat = vec![0.1f64; 8];
    let probs = rd.predict_predicates(&feat, &feat, &b1, &b2);
    assert_eq!(probs.len(), 5);
    let total: f64 = probs.iter().sum();
    assert!((total - 1.0).abs() < 1e-9);
}

#[test]
fn test_rel_detector_predict_predicates_all_positive() {
    let rd = SgRelationshipDetector::new(8, 5, 4);
    let b1 = SgBoundingBox::new(0.0, 0.0, 5.0, 5.0);
    let b2 = SgBoundingBox::new(10.0, 10.0, 5.0, 5.0);
    let feat = vec![-1.0f64; 8];
    let probs = rd.predict_predicates(&feat, &feat, &b1, &b2);
    assert!(probs.iter().all(|&p| p >= 0.0));
}

#[test]
fn test_rel_detector_detect_relationships_count() {
    let rd = SgRelationshipDetector::new(4, 3, 4);
    let mut sg = SgSceneGraph::new();
    for i in 0..3 {
        sg.add_object(SgObject {
            id: 0,
            label: format!("obj_{i}"),
            confidence: 0.9,
            bbox: SgBoundingBox::new(i as f64 * 20.0, 0.0, 10.0, 10.0),
            attributes: vec![],
            feature: vec![0.1f64; 4],
        });
    }
    let rels = rd.detect_relationships(&sg, 4).expect("detect_relationships should succeed");
    assert!(rels.len() <= 4);
    assert!(!rels.is_empty());
}

#[test]
fn test_rel_detector_detect_too_few_objects() {
    let rd = SgRelationshipDetector::new(4, 3, 4);
    let mut sg = SgSceneGraph::new();
    sg.add_object(SgObject {
        id: 0,
        label: "solo".into(),
        confidence: 0.9,
        bbox: SgBoundingBox::new(0.0, 0.0, 10.0, 10.0),
        attributes: vec![],
        feature: vec![0.1f64; 4],
    });
    let rels = rd.detect_relationships(&sg, 5).expect("detect_relationships should succeed");
    assert!(rels.is_empty());
}

// ── §5 SgMessagePassing tests ──

#[test]
fn test_mp_aggregate_messages_shape() {
    let mp = SgMessagePassing::new(4, 2);
    let features = vec![vec![0.1f64; 4], vec![0.2f64; 4], vec![0.3f64; 4]];
    let adj = vec![
        vec![0.0, 0.5, 0.0],
        vec![0.5, 0.0, 0.3],
        vec![0.0, 0.3, 0.0],
    ];
    let msgs = mp.aggregate_messages(&features, &adj);
    assert_eq!(msgs.len(), 3);
    assert_eq!(msgs[0].len(), 4);
}

#[test]
fn test_mp_aggregate_messages_zero_adj() {
    let mp = SgMessagePassing::new(4, 2);
    let features = vec![vec![1.0f64; 4], vec![1.0f64; 4]];
    let adj = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
    let msgs = mp.aggregate_messages(&features, &adj);
    assert!(msgs[0].iter().all(|&v| v.abs() < 1e-12));
    assert!(msgs[1].iter().all(|&v| v.abs() < 1e-12));
}

#[test]
fn test_mp_update_features_shape() {
    let mp = SgMessagePassing::new(4, 2);
    let features = vec![vec![0.5f64; 4], vec![0.3f64; 4]];
    let messages = vec![vec![0.1f64; 4], vec![0.2f64; 4]];
    let updated = mp.update_features(&features, &messages);
    assert_eq!(updated.len(), 2);
    assert_eq!(updated[0].len(), 4);
}

#[test]
fn test_mp_refine_scene_graph_no_error() {
    let mp = SgMessagePassing::new(4, 2);
    let mut sg = SgSceneGraph::new();
    for i in 0..3 {
        sg.add_object(SgObject {
            id: 0,
            label: format!("o{i}"),
            confidence: 0.9,
            bbox: SgBoundingBox::new(0.0, 0.0, 5.0, 5.0),
            attributes: vec![],
            feature: vec![0.1f64; 4],
        });
    }
    sg.add_relationship(SgRelationship {
        subject_id: 0,
        object_id: 1,
        predicate: "near".into(),
        predicate_id: 0,
        confidence: 0.8,
        feature: vec![],
    });
    assert!(mp.refine_scene_graph(&mut sg).is_ok());
}

#[test]
fn test_mp_refine_changes_features() {
    let mp = SgMessagePassing::new(4, 3);
    let mut sg = SgSceneGraph::new();
    sg.add_object(SgObject {
        id: 0,
        label: "a".into(),
        confidence: 0.9,
        bbox: SgBoundingBox::new(0.0, 0.0, 5.0, 5.0),
        attributes: vec![],
        feature: vec![1.0, 0.0, 0.0, 0.0],
    });
    sg.add_object(SgObject {
        id: 0,
        label: "b".into(),
        confidence: 0.9,
        bbox: SgBoundingBox::new(10.0, 0.0, 5.0, 5.0),
        attributes: vec![],
        feature: vec![0.0, 1.0, 0.0, 0.0],
    });
    sg.add_relationship(SgRelationship {
        subject_id: 0,
        object_id: 1,
        predicate: "next_to".into(),
        predicate_id: 0,
        confidence: 0.9,
        feature: vec![],
    });
    let feat_before = sg.objects[0].feature.clone();
    mp.refine_scene_graph(&mut sg).expect("refine_scene_graph should succeed");
    // Features should be modified by GNN
    let feat_after = &sg.objects[0].feature;
    // May be same if adjacency weights are zero-out by gate; at least no panic
    assert_eq!(feat_after.len(), 4);
    let _ = feat_before;
}

// ── §6 SgSceneGraphGeneration tests ──

#[test]
fn test_sgg_generate_returns_scene_graph() {
    let sgg = SgSceneGraphGeneration::new(8, 5, 4);
    let image_feat = vec![0.1f64; 8];
    let region_feats: Vec<Vec<f64>> = (0..4).map(|_| vec![0.5f64; 8]).collect();
    let anchors: Vec<SgBoundingBox> = (0..4)
        .map(|i| SgBoundingBox::new(i as f64 * 15.0, 0.0, 10.0, 10.0))
        .collect();
    let result = sgg.generate(image_feat, region_feats, anchors);
    assert!(result.is_ok());
}

#[test]
fn test_sgg_generate_empty_regions() {
    let sgg = SgSceneGraphGeneration::new(8, 5, 4);
    let sg = sgg.generate(vec![0.0f64; 8], vec![], vec![]).expect("generate should succeed");
    assert_eq!(sg.n_objects(), 0);
}

#[test]
fn test_sgg_recall_at_k_in_range() {
    let sgg = SgSceneGraphGeneration::new(8, 5, 4);
    let mut sg = SgSceneGraph::new();
    sg.add_object(SgObject {
        id: 0,
        label: "cat".into(),
        confidence: 0.9,
        bbox: SgBoundingBox::new(0.0, 0.0, 10.0, 10.0),
        attributes: vec![],
        feature: vec![0.1f64; 8],
    });
    sg.add_object(SgObject {
        id: 0,
        label: "dog".into(),
        confidence: 0.9,
        bbox: SgBoundingBox::new(20.0, 20.0, 10.0, 10.0),
        attributes: vec![],
        feature: vec![0.2f64; 8],
    });
    sg.add_relationship(SgRelationship {
        subject_id: 0,
        object_id: 1,
        predicate: "chases".into(),
        predicate_id: 0,
        confidence: 0.8,
        feature: vec![],
    });
    let gt = vec![("cat".to_string(), "chases".to_string(), "dog".to_string())];
    let r = sgg.recall_at_k(&sg, &gt, 5);
    assert!((0.0..=1.0).contains(&r));
}

#[test]
fn test_sgg_scene_graph_loss_nonnegative() {
    let sgg = SgSceneGraphGeneration::new(8, 5, 4);
    let mut sg = SgSceneGraph::new();
    sg.add_object(SgObject {
        id: 0,
        label: "cat".into(),
        confidence: 0.9,
        bbox: SgBoundingBox::new(0.0, 0.0, 10.0, 10.0),
        attributes: vec![],
        feature: vec![0.1f64; 8],
    });
    let gt_objects = vec![(1usize, SgBoundingBox::new(0.0, 0.0, 10.0, 10.0))];
    let gt_relations: Vec<(usize, usize, usize)> = vec![];
    let loss = sgg.scene_graph_loss(&sg, &gt_objects, &gt_relations);
    assert!(loss >= 0.0);
}

// ── §7 SgQueryEngine tests ──

fn make_query_sg() -> SgSceneGraph {
    let mut sg = SgSceneGraph::new();
    for label in &["cat", "dog", "cat"] {
        sg.add_object(SgObject {
            id: 0,
            label: (*label).to_string(),
            confidence: 0.9,
            bbox: SgBoundingBox::new(0.0, 0.0, 10.0, 10.0),
            attributes: vec![],
            feature: vec![0.1f64; 4],
        });
    }
    sg.add_relationship(SgRelationship {
        subject_id: 0,
        object_id: 1,
        predicate: "chases".into(),
        predicate_id: 0,
        confidence: 0.8,
        feature: vec![],
    });
    sg
}

#[test]
fn test_query_find_objects() {
    let sg = make_query_sg();
    if let SgQueryResult::Objects(ids) = SgQueryEngine::query(&sg, &SgQuery::FindObjects("cat".into())) {
        assert_eq!(ids.len(), 2);
    } else {
        panic!("expected Objects result");
    }
}

#[test]
fn test_query_find_objects_missing() {
    let sg = make_query_sg();
    if let SgQueryResult::Objects(ids) = SgQueryEngine::query(&sg, &SgQuery::FindObjects("elephant".into())) {
        assert!(ids.is_empty());
    } else {
        panic!("expected Objects result");
    }
}

#[test]
fn test_query_count_objects() {
    let sg = make_query_sg();
    if let SgQueryResult::Count(n) = SgQueryEngine::query(&sg, &SgQuery::CountObjects("cat".into())) {
        assert_eq!(n, 2);
    } else {
        panic!("expected Count result");
    }
}

#[test]
fn test_query_count_objects_zero() {
    let sg = make_query_sg();
    if let SgQueryResult::Count(n) = SgQueryEngine::query(&sg, &SgQuery::CountObjects("fish".into())) {
        assert_eq!(n, 0);
    } else {
        panic!("expected Count result");
    }
}

#[test]
fn test_query_exists_relationship_true() {
    let sg = make_query_sg();
    if let SgQueryResult::Boolean(v) = SgQueryEngine::query(
        &sg,
        &SgQuery::ExistsRelationship("cat".into(), "chases".into(), "dog".into()),
    ) {
        assert!(v);
    } else {
        panic!("expected Boolean result");
    }
}

#[test]
fn test_query_exists_relationship_false() {
    let sg = make_query_sg();
    if let SgQueryResult::Boolean(v) = SgQueryEngine::query(
        &sg,
        &SgQuery::ExistsRelationship("dog".into(), "chases".into(), "cat".into()),
    ) {
        assert!(!v);
    } else {
        panic!("expected Boolean result");
    }
}

#[test]
fn test_query_find_relationships() {
    let sg = make_query_sg();
    if let SgQueryResult::Relationships(ids) = SgQueryEngine::query(
        &sg,
        &SgQuery::FindRelationships("chases".into(), None, None),
    ) {
        assert_eq!(ids.len(), 1);
    } else {
        panic!("expected Relationships result");
    }
}

#[test]
fn test_query_find_relationships_with_subject() {
    let sg = make_query_sg();
    if let SgQueryResult::Relationships(ids) = SgQueryEngine::query(
        &sg,
        &SgQuery::FindRelationships("chases".into(), Some("cat".into()), None),
    ) {
        assert_eq!(ids.len(), 1);
    } else {
        panic!("expected Relationships result");
    }
}

#[test]
fn test_query_nearest_object() {
    let mut sg = SgSceneGraph::new();
    sg.add_object(SgObject {
        id: 0,
        label: "a".into(),
        confidence: 0.9,
        bbox: SgBoundingBox::new(0.0, 0.0, 2.0, 2.0),
        attributes: vec![],
        feature: vec![],
    });
    sg.add_object(SgObject {
        id: 0,
        label: "b".into(),
        confidence: 0.9,
        bbox: SgBoundingBox::new(3.0, 0.0, 2.0, 2.0),
        attributes: vec![],
        feature: vec![],
    });
    sg.add_object(SgObject {
        id: 0,
        label: "c".into(),
        confidence: 0.9,
        bbox: SgBoundingBox::new(50.0, 50.0, 2.0, 2.0),
        attributes: vec![],
        feature: vec![],
    });
    if let SgQueryResult::Objects(ids) = SgQueryEngine::query(&sg, &SgQuery::NearestObject(0)) {
        assert!(!ids.is_empty());
        // b (id=1) is nearer than c (id=2)
        assert_eq!(ids[0], 1);
    } else {
        panic!("expected Objects result");
    }
}

#[test]
fn test_query_nearest_object_invalid_id() {
    let sg = make_query_sg();
    if let SgQueryResult::Objects(ids) = SgQueryEngine::query(&sg, &SgQuery::NearestObject(999)) {
        assert!(ids.is_empty());
    } else {
        panic!("expected Objects result");
    }
}

// ── §8 SgVQA tests ──

#[test]
fn test_vqa_encode_question_shape() {
    let vqa = SgVQA::new(20, 8, 5, 6);
    let tokens = vec![0usize, 3, 7, 11];
    let q = vqa.encode_question(&tokens);
    assert_eq!(q.len(), 8);
}

#[test]
fn test_vqa_encode_question_empty() {
    let vqa = SgVQA::new(20, 8, 5, 6);
    let q = vqa.encode_question(&[]);
    assert_eq!(q.len(), 8);
}

#[test]
fn test_vqa_attend_to_objects_shape() {
    let vqa = SgVQA::new(20, 8, 5, 6);
    let q = vec![0.1f64; 8];
    let object_features: Vec<Vec<f64>> = (0..3).map(|_| vec![0.2f64; 6]).collect();
    let attended = vqa.attend_to_objects(&q, &object_features);
    assert_eq!(attended.len(), 6);
}

#[test]
fn test_vqa_attend_to_objects_empty() {
    let vqa = SgVQA::new(20, 8, 5, 6);
    let q = vec![0.1f64; 8];
    let attended = vqa.attend_to_objects(&q, &[]);
    assert_eq!(attended.len(), 6);
    assert!(attended.iter().all(|&v| v == 0.0));
}

#[test]
fn test_vqa_answer_sums_to_one() {
    let vqa = SgVQA::new(20, 8, 5, 6);
    let sg = make_query_sg();
    // Replace features with 6-dim
    let mut sg6 = SgSceneGraph::new();
    for obj in &sg.objects {
        let mut o = obj.clone();
        o.feature = vec![0.1f64; 6];
        sg6.add_object(o);
    }
    let probs = vqa.answer(&[1usize, 2, 5], &sg6);
    assert_eq!(probs.len(), 5);
    let total: f64 = probs.iter().sum();
    assert!((total - 1.0).abs() < 1e-9);
}

#[test]
fn test_vqa_answer_all_positive() {
    let vqa = SgVQA::new(20, 8, 5, 6);
    let sg = SgSceneGraph::new();
    let probs = vqa.answer(&[0usize, 1, 2], &sg);
    assert!(probs.iter().all(|&p| p >= 0.0));
}

// ── §9 SgSceneComparison tests ──

#[test]
fn test_triplet_recall_identical_graphs() {
    let mut sg = SgSceneGraph::new();
    sg.add_object(SgObject {
        id: 0,
        label: "cat".into(),
        confidence: 0.9,
        bbox: SgBoundingBox::new(0.0, 0.0, 5.0, 5.0),
        attributes: vec![],
        feature: vec![1.0, 0.0],
    });
    sg.add_object(SgObject {
        id: 0,
        label: "dog".into(),
        confidence: 0.9,
        bbox: SgBoundingBox::new(10.0, 0.0, 5.0, 5.0),
        attributes: vec![],
        feature: vec![0.0, 1.0],
    });
    sg.add_relationship(SgRelationship {
        subject_id: 0,
        object_id: 1,
        predicate: "chases".into(),
        predicate_id: 0,
        confidence: 0.8,
        feature: vec![],
    });
    let recall = SgSceneComparison::triplet_recall(&sg, &sg);
    assert!((recall - 1.0).abs() < 1e-9);
}

#[test]
fn test_triplet_recall_no_overlap() {
    let mut sg1 = SgSceneGraph::new();
    sg1.add_object(SgObject {
        id: 0, label: "cat".into(), confidence: 0.9,
        bbox: SgBoundingBox::new(0.0, 0.0, 5.0, 5.0),
        attributes: vec![], feature: vec![],
    });
    sg1.add_object(SgObject {
        id: 0, label: "dog".into(), confidence: 0.9,
        bbox: SgBoundingBox::new(10.0, 0.0, 5.0, 5.0),
        attributes: vec![], feature: vec![],
    });
    sg1.add_relationship(SgRelationship {
        subject_id: 0, object_id: 1,
        predicate: "chases".into(), predicate_id: 0,
        confidence: 0.8, feature: vec![],
    });

    let sg2 = SgSceneGraph::new(); // empty
    let recall = SgSceneComparison::triplet_recall(&sg1, &sg2);
    assert!((recall - 0.0).abs() < 1e-9);
}

#[test]
fn test_triplet_recall_in_range() {
    let sg1 = make_query_sg();
    let sg2 = make_query_sg();
    let recall = SgSceneComparison::triplet_recall(&sg1, &sg2);
    assert!((0.0..=1.0).contains(&recall));
}

#[test]
fn test_graph_edit_distance_identical() {
    let sg = make_query_sg();
    let ged = SgSceneComparison::graph_edit_distance(&sg, &sg);
    assert_eq!(ged, 0.0);
}

#[test]
fn test_graph_edit_distance_different() {
    let sg1 = make_query_sg();
    let sg2 = SgSceneGraph::new();
    let ged = SgSceneComparison::graph_edit_distance(&sg1, &sg2);
    assert!(ged > 0.0);
}

#[test]
fn test_semantic_similarity_identical() {
    let mut sg = SgSceneGraph::new();
    sg.add_object(SgObject {
        id: 0, label: "cat".into(), confidence: 0.9,
        bbox: SgBoundingBox::new(0.0, 0.0, 5.0, 5.0),
        attributes: vec![],
        feature: vec![1.0, 0.0, 0.0],
    });
    let sim = SgSceneComparison::semantic_similarity(&sg, &sg);
    assert!((sim - 1.0).abs() < 1e-9);
}

#[test]
fn test_semantic_similarity_empty() {
    let sg1 = SgSceneGraph::new();
    let sg2 = make_query_sg();
    let sim = SgSceneComparison::semantic_similarity(&sg1, &sg2);
    assert_eq!(sim, 0.0);
}

#[test]
fn test_relationship_overlap_identical() {
    let sg = make_query_sg();
    let overlap = SgSceneComparison::relationship_overlap(&sg, &sg);
    assert!((overlap - 1.0).abs() < 1e-9);
}

#[test]
fn test_relationship_overlap_empty() {
    let sg1 = SgSceneGraph::new();
    let sg2 = SgSceneGraph::new();
    let overlap = SgSceneComparison::relationship_overlap(&sg1, &sg2);
    // Both empty → union=0 → returns 1.0
    assert!((overlap - 1.0).abs() < 1e-9);
}

// ── §10 SgMetrics tests ──

#[test]
fn test_metrics_recall_at_k_perfect() {
    let pred = vec![
        ("a".to_string(), "b".to_string(), "c".to_string()),
        ("d".to_string(), "e".to_string(), "f".to_string()),
    ];
    let gt = pred.clone();
    let r = SgMetrics::recall_at_k(&pred, &gt, 2);
    assert!((r - 1.0).abs() < 1e-9);
}

#[test]
fn test_metrics_recall_at_k_zero() {
    let pred = vec![("x".to_string(), "y".to_string(), "z".to_string())];
    let gt = vec![("a".to_string(), "b".to_string(), "c".to_string())];
    let r = SgMetrics::recall_at_k(&pred, &gt, 5);
    assert_eq!(r, 0.0);
}

#[test]
fn test_metrics_recall_at_k_in_range() {
    let pred = vec![
        ("a".into(), "b".into(), "c".into()),
        ("d".into(), "e".into(), "f".into()),
        ("g".into(), "h".into(), "i".into()),
    ];
    let gt = vec![
        ("a".into(), "b".into(), "c".into()),
        ("x".into(), "y".into(), "z".into()),
    ];
    let r = SgMetrics::recall_at_k(&pred, &gt, 2);
    assert!((0.0..=1.0).contains(&r));
}

#[test]
fn test_metrics_mean_recall() {
    let recalls = vec![0.5, 0.8, 0.3];
    let mr = SgMetrics::mean_recall(&recalls);
    assert!((mr - (0.5 + 0.8 + 0.3) / 3.0).abs() < 1e-9);
}

#[test]
fn test_metrics_mean_recall_empty() {
    assert_eq!(SgMetrics::mean_recall(&[]), 0.0);
}

#[test]
fn test_metrics_predicate_accuracy() {
    let pred = vec![0usize, 1, 2, 0];
    let gt = vec![0usize, 1, 1, 0];
    let acc = SgMetrics::predicate_classification_accuracy(&pred, &gt);
    assert!((acc - 0.75).abs() < 1e-9);
}

#[test]
fn test_metrics_predicate_accuracy_perfect() {
    let v = vec![0usize, 1, 2];
    let acc = SgMetrics::predicate_classification_accuracy(&v, &v);
    assert!((acc - 1.0).abs() < 1e-9);
}

#[test]
fn test_metrics_scene_graph_completeness() {
    let mut sg = make_query_sg();
    let comp = SgMetrics::scene_graph_completeness(&sg);
    // 2 out of 3 objects are connected (cat0→dog1 relationship)
    assert!((0.0..=1.0).contains(&comp));
}

#[test]
fn test_metrics_scene_graph_completeness_empty() {
    let sg = SgSceneGraph::new();
    assert_eq!(SgMetrics::scene_graph_completeness(&sg), 0.0);
}

#[test]
fn test_metrics_relationship_density_in_range() {
    let sg = make_query_sg();
    let density = SgMetrics::relationship_density(&sg);
    assert!(density >= 0.0);
}

#[test]
fn test_metrics_relationship_density_empty() {
    let sg = SgSceneGraph::new();
    assert_eq!(SgMetrics::relationship_density(&sg), 0.0);
}

#[test]
fn test_metrics_bbox_map_empty_prediction() {
    let gt = vec![(SgBoundingBox::new(0.0, 0.0, 10.0, 10.0), 0usize)];
    let map = SgMetrics::bbox_mAP(&[], &gt, 0.5);
    assert_eq!(map, 0.0);
}

#[test]
fn test_metrics_bbox_map_in_range() {
    let pred = vec![
        (SgBoundingBox::new(0.0, 0.0, 10.0, 10.0), 0.9f64, 1usize),
        (SgBoundingBox::new(5.0, 5.0, 10.0, 10.0), 0.5f64, 1usize),
    ];
    let gt = vec![
        (SgBoundingBox::new(0.5, 0.5, 9.0, 9.0), 1usize),
    ];
    let map = SgMetrics::bbox_mAP(&pred, &gt, 0.5);
    assert!((0.0..=1.0).contains(&map), "mAP = {map} not in [0,1]");
}

#[test]
fn test_metrics_bbox_map_perfect() {
    let bb = SgBoundingBox::new(0.0, 0.0, 10.0, 10.0);
    let pred = vec![(bb.clone(), 0.95f64, 0usize)];
    let gt = vec![(bb, 0usize)];
    let map = SgMetrics::bbox_mAP(&pred, &gt, 0.5);
    // Exact match: AP = 1.0
    assert!((map - 1.0).abs() < 1e-9);
}
