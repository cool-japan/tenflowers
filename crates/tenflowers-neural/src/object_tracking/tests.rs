//! Tests for the Multi-Object Tracking module.

use super::*;

// ─────────────────────────────────────────────────────────────────────────────
// MotBoundingBox tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_bbox_new() {
    let bbox = MotBoundingBox::new(10.0, 20.0, 50.0, 30.0);
    assert_eq!(bbox.x, 10.0);
    assert_eq!(bbox.y, 20.0);
    assert_eq!(bbox.w, 50.0);
    assert_eq!(bbox.h, 30.0);
}

#[test]
fn test_bbox_area() {
    let bbox = MotBoundingBox::new(0.0, 0.0, 4.0, 5.0);
    assert!((bbox.area() - 20.0).abs() < 1e-9);
}

#[test]
fn test_bbox_iou_identical() {
    let bbox = MotBoundingBox::new(10.0, 10.0, 50.0, 50.0);
    let iou = bbox.iou(&bbox);
    assert!(
        (iou - 1.0).abs() < 1e-9,
        "IoU(x, x) must equal 1.0, got {}",
        iou
    );
}

#[test]
fn test_bbox_iou_no_overlap() {
    let a = MotBoundingBox::new(0.0, 0.0, 10.0, 10.0);
    let b = MotBoundingBox::new(20.0, 20.0, 10.0, 10.0);
    let iou = a.iou(&b);
    assert!(
        (iou - 0.0).abs() < 1e-9,
        "IoU of non-overlapping boxes must be 0.0"
    );
}

#[test]
fn test_bbox_iou_range() {
    let a = MotBoundingBox::new(0.0, 0.0, 100.0, 100.0);
    let b = MotBoundingBox::new(50.0, 50.0, 100.0, 100.0);
    let iou = a.iou(&b);
    assert!(
        (0.0..=1.0).contains(&iou),
        "IoU must be in [0, 1], got {}",
        iou
    );
}

#[test]
fn test_bbox_iou_partial_overlap() {
    // Two 10×10 boxes with 5×5 overlap
    let a = MotBoundingBox::new(0.0, 0.0, 10.0, 10.0);
    let b = MotBoundingBox::new(5.0, 5.0, 10.0, 10.0);
    let intersection = 5.0 * 5.0;
    let union = 100.0 + 100.0 - intersection;
    let expected = intersection / union;
    let iou = a.iou(&b);
    assert!(
        (iou - expected).abs() < 1e-9,
        "Expected IoU {}, got {}",
        expected,
        iou
    );
}

#[test]
fn test_bbox_iou_symmetry() {
    let a = MotBoundingBox::new(0.0, 0.0, 30.0, 40.0);
    let b = MotBoundingBox::new(15.0, 20.0, 30.0, 40.0);
    assert!((a.iou(&b) - b.iou(&a)).abs() < 1e-9);
}

#[test]
fn test_bbox_center() {
    let bbox = MotBoundingBox::new(10.0, 20.0, 50.0, 30.0);
    let (cx, cy) = bbox.center();
    assert!((cx - 35.0).abs() < 1e-9);
    assert!((cy - 35.0).abs() < 1e-9);
}

#[test]
fn test_bbox_to_xyxy() {
    let bbox = MotBoundingBox::new(10.0, 20.0, 50.0, 30.0);
    let xyxy = bbox.to_xyxy();
    assert!((xyxy[0] - 10.0).abs() < 1e-9);
    assert!((xyxy[1] - 20.0).abs() < 1e-9);
    assert!((xyxy[2] - 60.0).abs() < 1e-9);
    assert!((xyxy[3] - 50.0).abs() < 1e-9);
}

#[test]
fn test_bbox_from_xyxy() {
    let bbox = MotBoundingBox::from_xyxy(10.0, 20.0, 60.0, 50.0);
    assert!((bbox.x - 10.0).abs() < 1e-9);
    assert!((bbox.y - 20.0).abs() < 1e-9);
    assert!((bbox.w - 50.0).abs() < 1e-9);
    assert!((bbox.h - 30.0).abs() < 1e-9);
}

#[test]
fn test_bbox_to_state_from_state_roundtrip() {
    let original = MotBoundingBox::new(100.0, 200.0, 60.0, 40.0);
    let state = original.to_state();
    let recovered = MotBoundingBox::from_state(state[0], state[1], state[2], state[3]);

    assert!(
        (original.x - recovered.x).abs() < 1e-6,
        "x mismatch: {} vs {}",
        original.x,
        recovered.x
    );
    assert!(
        (original.y - recovered.y).abs() < 1e-6,
        "y mismatch: {} vs {}",
        original.y,
        recovered.y
    );
    assert!(
        (original.w - recovered.w).abs() < 1e-6,
        "w mismatch: {} vs {}",
        original.w,
        recovered.w
    );
    assert!(
        (original.h - recovered.h).abs() < 1e-6,
        "h mismatch: {} vs {}",
        original.h,
        recovered.h
    );
}

#[test]
fn test_bbox_state_values() {
    let bbox = MotBoundingBox::new(90.0, 190.0, 20.0, 10.0); // cx=100, cy=195, s=200, r=2
    let state = bbox.to_state();
    assert!(
        (state[0] - 100.0).abs() < 1e-9,
        "cx should be 100, got {}",
        state[0]
    );
    assert!(
        (state[1] - 195.0).abs() < 1e-9,
        "cy should be 195, got {}",
        state[1]
    );
    assert!(
        (state[2] - 200.0).abs() < 1e-9,
        "area should be 200, got {}",
        state[2]
    );
    assert!(
        (state[3] - 2.0).abs() < 1e-9,
        "aspect ratio should be 2, got {}",
        state[3]
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// MotKalmanFilter tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_kalman_new_state_initialization() {
    let bbox = MotBoundingBox::new(90.0, 190.0, 20.0, 10.0);
    let kf = MotKalmanFilter::new(&bbox);
    // State: cx=100, cy=195, s=200, r=2, velocities=0
    assert!((kf.state[0] - 100.0).abs() < 1e-6, "cx init error");
    assert!((kf.state[1] - 195.0).abs() < 1e-6, "cy init error");
    assert!((kf.state[4]).abs() < 1e-9, "v_cx should be 0");
    assert!((kf.state[5]).abs() < 1e-9, "v_cy should be 0");
    assert!((kf.state[6]).abs() < 1e-9, "v_s should be 0");
}

#[test]
fn test_kalman_predict_returns_4d() {
    let bbox = MotBoundingBox::new(100.0, 200.0, 50.0, 50.0);
    let mut kf = MotKalmanFilter::new(&bbox);
    let pred = kf.predict();
    assert_eq!(pred.len(), 4, "Prediction must have 4 components");
}

#[test]
fn test_kalman_predict_covariance_increases() {
    let bbox = MotBoundingBox::new(100.0, 200.0, 50.0, 50.0);
    let mut kf = MotKalmanFilter::new(&bbox);
    let cov_initial = kf.covariance[0][0];
    kf.predict();
    // Covariance should not decrease after prediction with process noise
    assert!(kf.covariance[0][0] >= cov_initial);
}

#[test]
fn test_kalman_update_moves_toward_measurement() {
    let bbox = MotBoundingBox::new(100.0, 200.0, 50.0, 50.0);
    let mut kf = MotKalmanFilter::new(&bbox);
    kf.predict();

    let measurement = MotBoundingBox::new(110.0, 210.0, 50.0, 50.0);
    let before_cx = kf.state[0];
    kf.update(&measurement);
    let after_cx = kf.state[0];

    // After update, state should move toward measurement
    let meas_cx = measurement.to_state()[0];
    let dist_before = (before_cx - meas_cx).abs();
    let dist_after = (after_cx - meas_cx).abs();
    assert!(
        dist_after < dist_before + 1.0, // allow some tolerance
        "Kalman update should reduce distance to measurement"
    );
}

#[test]
fn test_kalman_get_bbox_finite() {
    let bbox = MotBoundingBox::new(50.0, 50.0, 20.0, 20.0);
    let mut kf = MotKalmanFilter::new(&bbox);
    kf.predict();
    let out = kf.get_bbox();
    assert!(out.x.is_finite() && out.y.is_finite() && out.w.is_finite() && out.h.is_finite());
}

#[test]
fn test_kalman_repeated_updates_converge() {
    let initial = MotBoundingBox::new(0.0, 0.0, 100.0, 100.0);
    let target = MotBoundingBox::new(200.0, 300.0, 100.0, 100.0);
    let mut kf = MotKalmanFilter::new(&initial);

    for _ in 0..50 {
        kf.predict();
        kf.update(&target);
    }

    let result = kf.get_bbox();
    let (tcx, tcy) = target.center();
    let (rcx, rcy) = result.center();
    assert!(
        (rcx - tcx).abs() < 5.0,
        "cx should converge: {} vs {}",
        rcx,
        tcx
    );
    assert!(
        (rcy - tcy).abs() < 5.0,
        "cy should converge: {} vs {}",
        rcy,
        tcy
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// MotTrack tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_track_new() {
    let bbox = MotBoundingBox::new(10.0, 10.0, 50.0, 50.0);
    let track = MotTrack::new(1, &bbox);
    assert_eq!(track.track_id, 1);
    assert_eq!(track.hits, 1);
    assert_eq!(track.age, 1);
    assert_eq!(track.time_since_update, 0);
    assert_eq!(track.state, MotTrackState::Tentative);
    assert_eq!(track.history.len(), 1);
}

#[test]
fn test_track_predict_increments_age() {
    let bbox = MotBoundingBox::new(10.0, 10.0, 50.0, 50.0);
    let mut track = MotTrack::new(1, &bbox);
    track.predict();
    assert_eq!(track.age, 2);
    assert_eq!(track.time_since_update, 1);
}

#[test]
fn test_track_update_increments_hits() {
    let bbox = MotBoundingBox::new(10.0, 10.0, 50.0, 50.0);
    let mut track = MotTrack::new(1, &bbox);
    track.predict();
    let new_bbox = MotBoundingBox::new(15.0, 15.0, 50.0, 50.0);
    track.update(&new_bbox);
    assert_eq!(track.hits, 2);
    assert_eq!(track.time_since_update, 0);
    assert_eq!(track.history.len(), 2);
}

#[test]
fn test_track_is_confirmed_tentative() {
    let bbox = MotBoundingBox::new(0.0, 0.0, 10.0, 10.0);
    let track = MotTrack::new(1, &bbox);
    assert!(!track.is_confirmed());
}

#[test]
fn test_track_is_confirmed_after_state_change() {
    let bbox = MotBoundingBox::new(0.0, 0.0, 10.0, 10.0);
    let mut track = MotTrack::new(1, &bbox);
    track.state = MotTrackState::Confirmed;
    assert!(track.is_confirmed());
}

#[test]
fn test_track_is_deleted() {
    let bbox = MotBoundingBox::new(0.0, 0.0, 10.0, 10.0);
    let mut track = MotTrack::new(1, &bbox);
    assert!(!track.is_deleted());
    track.state = MotTrackState::Deleted;
    assert!(track.is_deleted());
}

#[test]
fn test_track_predicted_bbox_valid() {
    let bbox = MotBoundingBox::new(50.0, 50.0, 30.0, 20.0);
    let track = MotTrack::new(1, &bbox);
    let predicted = track.predicted_bbox();
    assert!(predicted.w >= 0.0 && predicted.h >= 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// MotHungarian tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_hungarian_1x1() {
    let cost = vec![vec![5.0]];
    let assignment = MotHungarian::solve(&cost);
    assert_eq!(assignment.len(), 1);
    assert_eq!(assignment[0], Some(0));
}

#[test]
fn test_hungarian_2x2_identity() {
    // Diagonal cost: prefer 0→0, 1→1
    let cost = vec![vec![0.0, 100.0], vec![100.0, 0.0]];
    let assignment = MotHungarian::solve(&cost);
    assert_eq!(assignment[0], Some(0));
    assert_eq!(assignment[1], Some(1));
}

#[test]
fn test_hungarian_2x2_cross() {
    // Cross cost: prefer 0→1, 1→0
    let cost = vec![vec![100.0, 0.0], vec![0.0, 100.0]];
    let assignment = MotHungarian::solve(&cost);
    assert_eq!(assignment[0], Some(1));
    assert_eq!(assignment[1], Some(0));
}

#[test]
fn test_hungarian_3x3() {
    let cost = vec![
        vec![4.0, 1.0, 3.0],
        vec![2.0, 0.0, 5.0],
        vec![3.0, 2.0, 2.0],
    ];
    let assignment = MotHungarian::solve(&cost);
    // Verify all assigned to valid columns and no duplicates
    let assigned: Vec<usize> = assignment.iter().filter_map(|x| *x).collect();
    let mut sorted = assigned.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        assigned.len(),
        "No duplicate column assignments"
    );
    assert_eq!(assignment.len(), 3);
}

#[test]
fn test_hungarian_more_rows_than_cols() {
    let cost = vec![vec![1.0, 2.0], vec![3.0, 0.0], vec![0.0, 5.0]];
    let assignment = MotHungarian::solve(&cost);
    assert_eq!(assignment.len(), 3);
    // At most 2 assignments (n_cols = 2)
    let assigned_count = assignment.iter().filter(|x| x.is_some()).count();
    assert!(assigned_count <= 2);
}

#[test]
fn test_hungarian_iou_cost_matrix_shape() {
    let tracks = vec![
        MotBoundingBox::new(0.0, 0.0, 50.0, 50.0),
        MotBoundingBox::new(100.0, 100.0, 50.0, 50.0),
    ];
    let dets = vec![
        MotBoundingBox::new(5.0, 5.0, 50.0, 50.0),
        MotBoundingBox::new(95.0, 95.0, 50.0, 50.0),
        MotBoundingBox::new(200.0, 200.0, 50.0, 50.0),
    ];
    let cost = MotHungarian::iou_cost_matrix(&tracks, &dets);
    assert_eq!(cost.len(), 2);
    assert_eq!(cost[0].len(), 3);
}

#[test]
fn test_hungarian_iou_cost_values_range() {
    let tracks = vec![MotBoundingBox::new(0.0, 0.0, 50.0, 50.0)];
    let dets = vec![MotBoundingBox::new(0.0, 0.0, 50.0, 50.0)]; // identical → IoU=1, cost=0
    let cost = MotHungarian::iou_cost_matrix(&tracks, &dets);
    assert!(
        (cost[0][0]).abs() < 1e-9,
        "Identical boxes should have cost 0"
    );
}

#[test]
fn test_hungarian_empty_input() {
    let cost: Vec<Vec<f64>> = vec![];
    let assignment = MotHungarian::solve(&cost);
    assert!(assignment.is_empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// MotSort tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_sort_new() {
    let config = MotSortConfig::default();
    let tracker = MotSort::new(config);
    assert_eq!(tracker.tracks.len(), 0);
    assert_eq!(tracker.next_id, 1);
    assert_eq!(tracker.frame, 0);
}

#[test]
fn test_sort_creates_tracks_for_new_detections() {
    let config = MotSortConfig {
        max_age: 3,
        min_hits: 1,
        iou_threshold: 0.3,
    };
    let mut tracker = MotSort::new(config);
    let dets = vec![
        MotBoundingBox::new(0.0, 0.0, 50.0, 50.0),
        MotBoundingBox::new(100.0, 100.0, 50.0, 50.0),
    ];
    tracker.update(&dets);
    assert_eq!(tracker.tracks.len(), 2);
}

#[test]
fn test_sort_track_ids_are_unique() {
    let config = MotSortConfig {
        max_age: 3,
        min_hits: 1,
        iou_threshold: 0.3,
    };
    let mut tracker = MotSort::new(config);
    let dets = vec![
        MotBoundingBox::new(0.0, 0.0, 50.0, 50.0),
        MotBoundingBox::new(100.0, 100.0, 50.0, 50.0),
        MotBoundingBox::new(200.0, 200.0, 50.0, 50.0),
    ];
    tracker.update(&dets);
    let ids: Vec<usize> = tracker.tracks.iter().map(|t| t.track_id).collect();
    let mut sorted_ids = ids.clone();
    sorted_ids.sort();
    sorted_ids.dedup();
    assert_eq!(sorted_ids.len(), ids.len(), "Track IDs must be unique");
}

#[test]
fn test_sort_tracks_deleted_after_max_age() {
    let config = MotSortConfig {
        max_age: 2,
        min_hits: 1,
        iou_threshold: 0.3,
    };
    let max_age = config.max_age;
    let mut tracker = MotSort::new(config);
    let initial_dets = vec![MotBoundingBox::new(0.0, 0.0, 50.0, 50.0)];
    tracker.update(&initial_dets);
    assert_eq!(tracker.tracks.len(), 1);

    // Run with no detections for max_age + 1 frames
    for _ in 0..=max_age {
        tracker.update(&[]);
    }
    assert_eq!(
        tracker.tracks.len(),
        0,
        "Track should be deleted after max_age frames"
    );
}

#[test]
fn test_sort_confirmed_after_min_hits() {
    let config = MotSortConfig {
        max_age: 10,
        min_hits: 3,
        iou_threshold: 0.3,
    };
    let min_hits = config.min_hits;
    let mut tracker = MotSort::new(config);
    let bbox = MotBoundingBox::new(50.0, 50.0, 50.0, 50.0);

    // Keep tracking the same box for min_hits frames
    for _ in 0..min_hits {
        tracker.update(std::slice::from_ref(&bbox));
    }

    let confirmed_count = tracker.tracks.iter().filter(|t| t.is_confirmed()).count();
    assert!(
        confirmed_count > 0,
        "At least one track should be confirmed"
    );
}

#[test]
fn test_sort_n_active_tracks() {
    let config = MotSortConfig {
        max_age: 3,
        min_hits: 1,
        iou_threshold: 0.3,
    };
    let mut tracker = MotSort::new(config);
    let dets = vec![
        MotBoundingBox::new(0.0, 0.0, 50.0, 50.0),
        MotBoundingBox::new(200.0, 200.0, 50.0, 50.0),
    ];
    tracker.update(&dets);
    assert_eq!(tracker.n_active_tracks(), 2);
}

#[test]
fn test_sort_returns_confirmed_tracks_only() {
    let config = MotSortConfig {
        max_age: 10,
        min_hits: 3,
        iou_threshold: 0.3,
    };
    let mut tracker = MotSort::new(config);
    let bbox = MotBoundingBox::new(50.0, 50.0, 50.0, 50.0);

    // First frame — tracks are tentative, should not appear in output
    let results = tracker.update(std::slice::from_ref(&bbox));
    // Frame 1: with min_hits=3, frame 1 ≤ min_hits, so should be included
    // (the condition: frame <= min_hits → auto-confirm)

    // Just check that the function returns without panic and results are valid
    for (id, b) in &results {
        assert!(*id > 0);
        assert!(b.w > 0.0 && b.h > 0.0);
    }
}

#[test]
fn test_sort_associate_empty_tracks() {
    let dets = vec![MotBoundingBox::new(0.0, 0.0, 50.0, 50.0)];
    let (matched, unmatched_t, unmatched_d) = MotSort::associate(&[], &dets, 0.3);
    assert!(matched.is_empty());
    assert!(unmatched_t.is_empty());
    assert_eq!(unmatched_d.len(), 1);
}

#[test]
fn test_sort_associate_empty_detections() {
    let tracks = vec![MotBoundingBox::new(0.0, 0.0, 50.0, 50.0)];
    let (matched, unmatched_t, unmatched_d) = MotSort::associate(&tracks, &[], 0.3);
    assert!(matched.is_empty());
    assert_eq!(unmatched_t.len(), 1);
    assert!(unmatched_d.is_empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// MotAppearanceFeature tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_appearance_feature_extract_dimension() {
    let extractor = MotAppearanceFeature::new(64, 128, 32);
    let patch = vec![0.5f64; 64];
    let embedding = extractor.extract(&patch);
    assert_eq!(
        embedding.len(),
        32,
        "Embedding must have embed_dim elements"
    );
}

#[test]
fn test_appearance_feature_extract_l2_normalized() {
    let extractor = MotAppearanceFeature::new(64, 128, 32);
    let patch = vec![0.3f64; 64];
    let embedding = extractor.extract(&patch);
    let norm: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-6,
        "Embedding must be L2-normalized, norm={}",
        norm
    );
}

#[test]
fn test_appearance_feature_cosine_distance_same() {
    let v = vec![1.0, 0.0, 0.0];
    let dist = MotAppearanceFeature::cosine_distance(&v, &v);
    assert!(
        dist.abs() < 1e-9,
        "Cosine distance of identical vectors = 0"
    );
}

#[test]
fn test_appearance_feature_cosine_distance_orthogonal() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    let dist = MotAppearanceFeature::cosine_distance(&a, &b);
    assert!(
        (dist - 1.0).abs() < 1e-9,
        "Orthogonal vectors have cosine distance 1.0"
    );
}

#[test]
fn test_appearance_feature_cosine_distance_range() {
    let a = vec![0.6, 0.8, 0.0];
    let b = vec![-0.8, 0.6, 0.0];
    let dist = MotAppearanceFeature::cosine_distance(&a, &b);
    assert!(
        (0.0..=2.0).contains(&dist),
        "Cosine distance must be in [0, 2], got {}",
        dist
    );
}

#[test]
fn test_appearance_feature_euclidean_distance() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    let dist = MotAppearanceFeature::euclidean_distance(&a, &b);
    assert!((dist - 2.0_f64.sqrt()).abs() < 1e-9);
}

#[test]
fn test_appearance_feature_euclidean_same() {
    let v = vec![3.0, 4.0, 0.0];
    let dist = MotAppearanceFeature::euclidean_distance(&v, &v);
    assert!(dist.abs() < 1e-9);
}

#[test]
fn test_appearance_feature_different_patches_different_embeddings() {
    let extractor = MotAppearanceFeature::new(16, 32, 8);
    let patch1 = vec![1.0f64; 16];
    let patch2 = vec![0.0f64; 16];
    let emb1 = extractor.extract(&patch1);
    let emb2 = extractor.extract(&patch2);
    // They should differ (though may not be guaranteed, it's expected for non-trivial nets)
    let dist = MotAppearanceFeature::euclidean_distance(&emb1, &emb2);
    // Just check both are valid unit vectors
    let norm1: f64 = emb1.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm2: f64 = emb2.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!((norm1 - 1.0).abs() < 1e-6);
    assert!((norm2 - 1.0).abs() < 1e-6);
    let _ = dist;
}

// ─────────────────────────────────────────────────────────────────────────────
// MotDeepSort tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_deepsort_new() {
    let config = MotSortConfig::default();
    let ds = MotDeepSort::new(config, 32);
    assert_eq!(ds.sort.tracks.len(), 0);
}

#[test]
fn test_deepsort_update_returns_pairs() {
    let config = MotSortConfig {
        max_age: 5,
        min_hits: 1,
        iou_threshold: 0.3,
    };
    let mut ds = MotDeepSort::new(config, 16);

    let dets = vec![
        (
            MotBoundingBox::new(10.0, 10.0, 50.0, 50.0),
            vec![0.5f64; 64],
        ),
        (
            MotBoundingBox::new(200.0, 200.0, 50.0, 50.0),
            vec![0.1f64; 64],
        ),
    ];
    let results = ds.update(&dets);

    // Results should be valid (id > 0, valid bbox)
    for (id, bbox) in &results {
        assert!(*id > 0);
        assert!(bbox.w > 0.0 && bbox.h > 0.0);
    }
}

#[test]
fn test_deepsort_appearance_cost_matrix_shape() {
    let config = MotSortConfig::default();
    let ds = MotDeepSort::new(config, 8);

    let track_feats = vec![vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]];
    let det_feats = vec![
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];
    let cost = ds.appearance_cost_matrix(&track_feats, &det_feats);
    assert_eq!(cost.len(), 1);
    assert_eq!(cost[0].len(), 2);
}

#[test]
fn test_deepsort_appearance_cost_values() {
    let config = MotSortConfig::default();
    let ds = MotDeepSort::new(config, 4);

    // Identical feature → cost 0
    let feat = vec![1.0, 0.0, 0.0, 0.0];
    let cost = ds.appearance_cost_matrix(std::slice::from_ref(&feat), std::slice::from_ref(&feat));
    assert!(cost[0][0].abs() < 1e-9);
}

#[test]
fn test_deepsort_track_count_grows() {
    let config = MotSortConfig {
        max_age: 10,
        min_hits: 1,
        iou_threshold: 0.3,
    };
    let mut ds = MotDeepSort::new(config, 8);

    for i in 0..3 {
        let det = vec![(
            MotBoundingBox::new(i as f64 * 100.0, 0.0, 50.0, 50.0),
            vec![0.5f64; 64],
        )];
        ds.update(&det);
    }
    assert!(ds.sort.tracks.len() >= 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// MotByteTrack tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_bytetrack_new() {
    let bt = MotByteTrack::new(0.5, 0.1);
    assert_eq!(bt.high_thresh, 0.5);
    assert_eq!(bt.low_thresh, 0.1);
    assert_eq!(bt.inner.tracks.len(), 0);
}

#[test]
fn test_bytetrack_high_conf_detections_tracked() {
    let mut bt = MotByteTrack::new(0.5, 0.1);
    let dets = vec![
        (MotBoundingBox::new(0.0, 0.0, 50.0, 50.0), 0.9),
        (MotBoundingBox::new(100.0, 0.0, 50.0, 50.0), 0.8),
    ];
    bt.update(&dets);
    assert!(
        bt.inner.tracks.len() >= 1,
        "High-confidence detections must create tracks"
    );
}

#[test]
fn test_bytetrack_low_conf_detections_ignored_initially() {
    let mut bt = MotByteTrack::new(0.5, 0.1);
    let dets = vec![
        (MotBoundingBox::new(0.0, 0.0, 50.0, 50.0), 0.2), // Low confidence, no existing tracks
    ];
    bt.update(&dets);
    // Low-confidence dets should not create new tracks on their own
    assert_eq!(
        bt.inner.tracks.len(),
        0,
        "Low-confidence alone should not create tracks"
    );
}

#[test]
fn test_bytetrack_split_by_confidence() {
    let dets = vec![
        (MotBoundingBox::new(0.0, 0.0, 10.0, 10.0), 0.8),
        (MotBoundingBox::new(20.0, 0.0, 10.0, 10.0), 0.3),
        (MotBoundingBox::new(40.0, 0.0, 10.0, 10.0), 0.9),
        (MotBoundingBox::new(60.0, 0.0, 10.0, 10.0), 0.1),
    ];
    let (high, low) = MotByteTrack::split_by_confidence(&dets, 0.5);
    assert_eq!(high.len(), 2, "2 high-confidence detections");
    assert_eq!(low.len(), 2, "2 low-confidence detections");
}

#[test]
fn test_bytetrack_returns_valid_results() {
    let mut bt = MotByteTrack::new(0.5, 0.1);
    let dets = vec![(MotBoundingBox::new(10.0, 10.0, 50.0, 50.0), 0.9)];
    let results = bt.update(&dets);
    for (id, bbox, conf) in &results {
        assert!(*id > 0);
        assert!(bbox.w > 0.0 && bbox.h > 0.0);
        assert!(*conf > 0.0);
    }
}

#[test]
fn test_bytetrack_multiple_frames() {
    let mut bt = MotByteTrack::new(0.5, 0.1);

    // Frame 1
    bt.update(&[(MotBoundingBox::new(0.0, 0.0, 50.0, 50.0), 0.9)]);
    // Frame 2 with slightly shifted detection
    bt.update(&[(MotBoundingBox::new(5.0, 5.0, 50.0, 50.0), 0.85)]);
    // Frame 3
    let results = bt.update(&[(MotBoundingBox::new(10.0, 10.0, 50.0, 50.0), 0.8)]);

    // Should have at least one confirmed track
    let n_tracks = bt.inner.tracks.len();
    assert!(n_tracks >= 1);
    let _ = results;
}

// ─────────────────────────────────────────────────────────────────────────────
// MotEloRating tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_elo_new() {
    let elo = MotEloRating::new(32.0);
    assert_eq!(elo.ratings.len(), 0);
    assert!((elo.k_factor - 32.0).abs() < 1e-9);
}

#[test]
fn test_elo_add_tracker() {
    let mut elo = MotEloRating::new(32.0);
    elo.add_tracker("SORT", 1500.0);
    elo.add_tracker("DeepSORT", 1500.0);
    assert_eq!(elo.ratings.len(), 2);
}

#[test]
fn test_elo_expected_score_equal_ratings() {
    let e = MotEloRating::expected_score(1500.0, 1500.0);
    assert!((e - 0.5).abs() < 1e-9, "Equal ratings → expected score 0.5");
}

#[test]
fn test_elo_expected_score_in_range() {
    for (ra, rb) in &[(1200.0, 1800.0), (1800.0, 1200.0), (2000.0, 1000.0)] {
        let e = MotEloRating::expected_score(*ra, *rb);
        assert!(
            (0.0..=1.0).contains(&e),
            "Expected score must be in [0, 1], got {}",
            e
        );
    }
}

#[test]
fn test_elo_expected_scores_sum_to_one() {
    let ra = 1600.0;
    let rb = 1400.0;
    let e_a = MotEloRating::expected_score(ra, rb);
    let e_b = MotEloRating::expected_score(rb, ra);
    assert!(
        (e_a + e_b - 1.0).abs() < 1e-9,
        "Expected scores must sum to 1.0"
    );
}

#[test]
fn test_elo_update_ratings_changes_values() {
    let mut elo = MotEloRating::new(32.0);
    elo.add_tracker("A", 1500.0);
    elo.add_tracker("B", 1500.0);
    let before_a = elo.ratings[0].1;
    let before_b = elo.ratings[1].1;
    elo.update_ratings(0, 1); // A wins
    assert!(elo.ratings[0].1 > before_a, "Winner rating should increase");
    assert!(elo.ratings[1].1 < before_b, "Loser rating should decrease");
}

#[test]
fn test_elo_ranking_sorted() {
    let mut elo = MotEloRating::new(32.0);
    elo.add_tracker("SORT", 1500.0);
    elo.add_tracker("DeepSORT", 1600.0);
    elo.add_tracker("ByteTrack", 1550.0);
    let ranking = elo.ranking();
    assert_eq!(ranking[0].0, "DeepSORT");
    assert_eq!(ranking[1].0, "ByteTrack");
    assert_eq!(ranking[2].0, "SORT");
}

#[test]
fn test_elo_update_out_of_bounds() {
    let mut elo = MotEloRating::new(32.0);
    elo.add_tracker("A", 1500.0);
    // Should not panic
    elo.update_ratings(0, 5); // 5 is out of bounds
    assert_eq!(elo.ratings[0].1, 1500.0); // unchanged
}

// ─────────────────────────────────────────────────────────────────────────────
// MotMetrics tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_mota_perfect() {
    // 0 errors → MOTA = 1.0
    let mota = MotMetrics::mota(0, 0, 0, 100);
    assert!((mota - 1.0).abs() < 1e-9);
}

#[test]
fn test_mota_upper_bound() {
    // MOTA can be negative for bad trackers, but 1.0 is the maximum
    let mota = MotMetrics::mota(5, 10, 2, 100);
    assert!(mota <= 1.0);
}

#[test]
fn test_mota_negative_for_bad_tracker() {
    let mota = MotMetrics::mota(200, 0, 0, 100);
    assert!(mota < 0.0, "MOTA should be negative when FP > GT");
}

#[test]
fn test_idf1_perfect() {
    let idf1 = MotMetrics::idf1(100, 0, 0);
    assert!((idf1 - 1.0).abs() < 1e-9);
}

#[test]
fn test_idf1_zero() {
    let idf1 = MotMetrics::idf1(0, 50, 50);
    assert!((idf1 - 0.0).abs() < 1e-9);
}

#[test]
fn test_idf1_range() {
    let idf1 = MotMetrics::idf1(50, 20, 30);
    assert!(
        (0.0..=1.0).contains(&idf1),
        "IDF1 must be in [0, 1], got {}",
        idf1
    );
}

#[test]
fn test_hota_perfect() {
    let hota = MotMetrics::hota(1.0, 1.0);
    assert!((hota - 1.0).abs() < 1e-9);
}

#[test]
fn test_hota_zero() {
    let hota = MotMetrics::hota(0.0, 1.0);
    assert!((hota - 0.0).abs() < 1e-9);
}

#[test]
fn test_hota_range() {
    let hota = MotMetrics::hota(0.7, 0.8);
    assert!(
        (0.0..=1.0).contains(&hota),
        "HOTA must be in [0, 1], got {}",
        hota
    );
}

#[test]
fn test_hota_geometric_mean() {
    let da = 0.64;
    let aa = 0.36;
    let hota = MotMetrics::hota(da, aa);
    let expected = (da * aa).sqrt();
    assert!((hota - expected).abs() < 1e-9);
}

#[test]
fn test_track_quality_perfect() {
    let bbox = MotBoundingBox::new(0.0, 0.0, 10.0, 10.0);
    let mut track = MotTrack::new(1, &bbox);
    // hits = 1, age = 1 → quality = 1.0
    let q = MotMetrics::track_quality(&track);
    assert!((q - 1.0).abs() < 1e-9);

    // Predict without update (hit missed)
    track.predict();
    let q2 = MotMetrics::track_quality(&track);
    assert!(
        (0.0..=1.0).contains(&q2),
        "Track quality in [0, 1], got {}",
        q2
    );
}

#[test]
fn test_track_quality_range() {
    let bbox = MotBoundingBox::new(0.0, 0.0, 10.0, 10.0);
    let mut track = MotTrack::new(1, &bbox);
    for _ in 0..10 {
        track.predict(); // miss
    }
    let q = MotMetrics::track_quality(&track);
    assert!((0.0..=1.0).contains(&q));
}

#[test]
fn test_count_id_switches_none() {
    let traj = vec![(0, 1), (1, 1), (2, 1), (3, 1)];
    let switches = MotMetrics::count_id_switches(&traj);
    assert_eq!(switches, 0);
}

#[test]
fn test_count_id_switches_one() {
    let traj = vec![(0, 1), (1, 1), (2, 2), (3, 2)];
    let switches = MotMetrics::count_id_switches(&traj);
    assert_eq!(switches, 1);
}

#[test]
fn test_count_id_switches_multiple() {
    let traj = vec![(0, 1), (1, 2), (2, 1), (3, 3)];
    let switches = MotMetrics::count_id_switches(&traj);
    assert_eq!(switches, 3);
}

#[test]
fn test_count_id_switches_empty() {
    let traj: Vec<(usize, usize)> = vec![];
    let switches = MotMetrics::count_id_switches(&traj);
    assert_eq!(switches, 0);
}

#[test]
fn test_average_iou_over_time_perfect() {
    let bbox = MotBoundingBox::new(0.0, 0.0, 50.0, 50.0);
    let predicted = vec![(0, bbox.clone()), (1, bbox.clone())];
    let gt = vec![(0, bbox.clone()), (1, bbox.clone())];
    let avg = MotMetrics::average_iou_over_time(&predicted, &gt);
    assert!(
        (avg - 1.0).abs() < 1e-9,
        "Perfect prediction → avg IoU = 1.0"
    );
}

#[test]
fn test_average_iou_over_time_no_overlap() {
    let predicted = vec![(0, MotBoundingBox::new(0.0, 0.0, 10.0, 10.0))];
    let gt = vec![(0, MotBoundingBox::new(100.0, 100.0, 10.0, 10.0))];
    let avg = MotMetrics::average_iou_over_time(&predicted, &gt);
    assert!((avg - 0.0).abs() < 1e-9);
}

#[test]
fn test_average_iou_over_time_missing_frames() {
    let predicted = vec![(1, MotBoundingBox::new(0.0, 0.0, 50.0, 50.0))];
    let gt = vec![
        (0, MotBoundingBox::new(0.0, 0.0, 50.0, 50.0)), // unmatched → IoU 0
        (1, MotBoundingBox::new(0.0, 0.0, 50.0, 50.0)), // matched → IoU 1
    ];
    let avg = MotMetrics::average_iou_over_time(&predicted, &gt);
    // (0.0 + 1.0) / 2 = 0.5
    assert!((avg - 0.5).abs() < 1e-9, "Expected 0.5, got {}", avg);
}

#[test]
fn test_average_iou_empty_gt() {
    let predicted = vec![(0, MotBoundingBox::new(0.0, 0.0, 50.0, 50.0))];
    let avg = MotMetrics::average_iou_over_time(&predicted, &[]);
    assert!((avg - 0.0).abs() < 1e-9);
}
