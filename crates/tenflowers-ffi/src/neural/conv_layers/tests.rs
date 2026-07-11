//! Tests for `conv_layers`: forward-value correctness (cross-checked against
//! the core ops directly) plus implicit-autograd integration (forward ->
//! loss(sum) -> backward -> gradient assertions, finite-difference verified
//! for Conv1D/Conv2D/Conv3D's weight/bias/input gradients, and
//! predecessor-gradient-flow verified for MaxPool2D/AvgPool2D).

use super::*;
use crate::neural::layers::PyDense;
use crate::neural::optimizers::PySGD;
use pyo3::Python;

fn make_tensor(data: Vec<f32>, shape: &[usize]) -> PyTensor {
    let tensor = Tensor::from_vec(data, shape).expect("tensor construction");
    PyTensor {
        tensor: Arc::new(tensor),
        requires_grad: false,
        is_pinned: false,
    }
}

/// Reset all thread-local implicit-autograd state so each test observes
/// a clean tape, mirroring `implicit_autograd::tests::reset_for_test`
/// (not reusable directly since it is private to that module).
fn reset_autograd_state() {
    // A fresh forward -> sum -> backward cycle on an unrelated, tiny
    // graph both exercises and then clears LEAVES/TRACKED_REGISTRY/the
    // tape via `run_backward`'s own cleanup - the only reset surface
    // available to this module, which does not have direct access to
    // implicit_autograd's private thread-locals.
    let x = make_tensor(vec![1.0], &[1]);
    let mut tracked = x.clone();
    tracked.requires_grad = true;
    implicit_autograd::mark_leaf(&tracked);
    let raw = tenflowers_core::ops::sum(&tracked.tensor, None, false)
        .expect("sum must succeed for reset");
    let scalar = PyTensor {
        tensor: Arc::new(raw),
        requires_grad: true,
        is_pinned: false,
    };
    let _ = implicit_autograd::record_and_link_unary(
        crate::implicit_autograd::UnaryOpKind::Sum {
            axes: None,
            keepdims: false,
        },
        &tracked,
        &scalar,
    );
    let _ = implicit_autograd::run_backward(&scalar);
}

#[test]
fn conv2d_forward_matches_core_op() {
    Python::initialize();
    Python::attach(|py| {
        reset_autograd_state();
        // in=2, out=3, kernel 2x2, valid padding, no bias.
        let conv = PyConv2D::new(py, 2, 3, (2, 2), None, None, None, None, Some(false))
            .expect("conv2d construction");
        // Distinct non-zero weights [out=3, in=2, 2, 2] = 24 elements.
        let weight_data: Vec<f32> = (1..=24).map(|v| v as f32 * 0.1).collect();
        let weight = Tensor::from_vec(weight_data, &[3, 2, 2, 2]).expect("weight");
        conv.weight_param
            .borrow(py)
            .set_data(weight.clone())
            .expect("set_data must succeed");

        // Input [1, 2, 4, 4] = 32 elements.
        let input_data: Vec<f32> = (1..=32).map(|v| v as f32).collect();
        let input = make_tensor(input_data, &[1, 2, 4, 4]);

        let out = conv.forward(py, &input).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 3, 3, 3]);

        let out_vec = out.tensor.to_vec().expect("out vec");
        assert!(
            out_vec.iter().any(|&v| v != 0.0),
            "output must not be all zeros"
        );

        // Cross-check against the core op invoked directly.
        let direct = tenflowers_core::ops::conv2d(
            input.tensor.as_ref(),
            &weight,
            None,
            (1, 1),
            "valid",
        )
        .expect("direct conv2d");
        let direct_vec = direct.to_vec().expect("direct vec");
        assert_eq!(out_vec.len(), direct_vec.len());
        for (a, b) in out_vec.iter().zip(direct_vec.iter()) {
            assert!((a - b).abs() < 1e-6, "ffi conv2d must equal core conv2d");
        }
    });
}

#[test]
fn conv2d_padding_changes_output_shape() {
    Python::initialize();
    Python::attach(|py| {
        reset_autograd_state();
        let conv = PyConv2D::new(py, 1, 1, (3, 3), None, Some((1, 1)), None, None, Some(false))
            .expect("conv2d construction");
        let weight = Tensor::from_vec(vec![1.0; 9], &[1, 1, 3, 3]).expect("weight");
        conv.weight_param.borrow(py).set_data(weight).expect("set_data");
        let input_data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
        let input = make_tensor(input_data, &[1, 1, 4, 4]);
        let out = conv.forward(py, &input).expect("forward");
        // "same"-style padding 1 with 3x3 kernel keeps spatial dims.
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 1, 4, 4]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        assert!(out_vec.iter().any(|&v| v != 0.0));
    });
}

#[test]
fn conv2d_groups_real_values() {
    Python::initialize();
    Python::attach(|py| {
        reset_autograd_state();
        // groups=2: output channel `oc` must only see its own group's input
        // channel. With in=2/out=2/groups=2 and 1x1 kernels, oc0 reads input
        // channel 0 and oc1 reads input channel 1 - no cross-group contamination.
        let conv = PyConv2D::new(py, 2, 2, (1, 1), None, None, None, Some(2), Some(false))
            .expect("conv2d construction");
        // Weight [out=2, in/groups=1, 1, 1]: oc0 scales by 10, oc1 scales by 100.
        let weight = Tensor::from_vec(vec![10.0, 100.0], &[2, 1, 1, 1]).expect("weight");
        conv.weight_param.borrow(py).set_data(weight).expect("set_data");
        // Input [1, 2, 2, 2]: channel 0 = 2.0, channel 1 = 3.0.
        let input = make_tensor(vec![2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0], &[1, 2, 2, 2]);
        let out = conv.forward(py, &input).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 2, 2, 2]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        // oc0 = 2*10 = 20 (would be 3*10 = 30 if groups were ignored); oc1 = 3*100 = 300.
        assert_eq!(
            out_vec,
            vec![20.0, 20.0, 20.0, 20.0, 300.0, 300.0, 300.0, 300.0]
        );
    });
}

#[test]
fn conv2d_dilation_real_values() {
    Python::initialize();
    Python::attach(|py| {
        reset_autograd_state();
        // 1 in / 1 out, kernel 2x2, dilation 2, valid, all-ones weight.
        let conv = PyConv2D::new(py, 1, 1, (2, 2), None, None, Some((2, 2)), None, Some(false))
            .expect("conv2d construction");
        let weight = Tensor::from_vec(vec![1.0; 4], &[1, 1, 2, 2]).expect("weight");
        conv.weight_param.borrow(py).set_data(weight).expect("set_data");
        // 4x4 input with values 1..16.
        let input_data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
        let input = make_tensor(input_data, &[1, 1, 4, 4]);
        let out = conv.forward(py, &input).expect("forward");
        // effective kernel = 3, out = (4-3)/1 + 1 = 2.
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 1, 2, 2]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        // (0,0)=1+3+9+11=24, (0,1)=2+4+10+12=28, (1,0)=5+7+13+15=40, (1,1)=6+8+14+16=44.
        assert_eq!(out_vec, vec![24.0, 28.0, 40.0, 44.0]);
    });
}

#[test]
fn conv2d_dilation_groups_padding_combined() {
    Python::initialize();
    Python::attach(|py| {
        reset_autograd_state();
        // Combined case: dilation + groups + explicit padding all at once.
        let conv = PyConv2D::new(
            py,
            2,
            2,
            (2, 2),
            None,
            Some((1, 1)),
            Some((2, 2)),
            Some(2),
            Some(false),
        )
        .expect("conv2d construction");
        // Weight [out=2, in/groups=1, 2, 2].
        let weight = Tensor::from_vec(vec![1.0; 8], &[2, 1, 2, 2]).expect("weight");
        conv.weight_param.borrow(py).set_data(weight).expect("set_data");
        // Input [1, 2, 4, 4].
        let input_data: Vec<f32> = (1..=32).map(|v| v as f32).collect();
        let input = make_tensor(input_data, &[1, 2, 4, 4]);
        let out = conv.forward(py, &input).expect("forward");
        // padded 6x6, effective kernel 3, out = (6-3)/1 + 1 = 4.
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 2, 4, 4]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        assert!(out_vec.iter().any(|&v| v != 0.0));
    });
}

#[test]
fn max_pool2d_forward_real_values() {
    let pool = PyMaxPool2D::new((2, 2), Some((2, 2)), None, None, None, None)
        .expect("maxpool construction");
    let input_data: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let input = make_tensor(input_data, &[1, 1, 4, 4]);
    let out = pool.forward(&input).expect("forward");
    assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 1, 2, 2]);
    let out_vec = out.tensor.to_vec().expect("out vec");
    assert_eq!(out_vec, vec![5.0, 7.0, 13.0, 15.0]);
}

#[test]
fn max_pool2d_explicit_padding_negative_input() {
    // Negative input proves the -inf fill (a 0.0 fill would wrongly win the max).
    let pool = PyMaxPool2D::new((2, 2), Some((2, 2)), Some((1, 1)), None, None, None)
        .expect("maxpool construction");
    let input = make_tensor(vec![-5.0; 4], &[1, 1, 2, 2]);
    let out = pool.forward(&input).expect("forward");
    // padded 4x4, out = (4-2)/2 + 1 = 2.
    assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 1, 2, 2]);
    let out_vec = out.tensor.to_vec().expect("out vec");
    // Each window sees exactly one real -5.0 (rest is -inf), so max = -5.0.
    assert_eq!(out_vec, vec![-5.0, -5.0, -5.0, -5.0]);
}

#[test]
fn max_pool2d_ceil_mode_real_values() {
    // 3x3 input, kernel 2, stride 2, ceil_mode=True -> 2x2 output.
    let pool = PyMaxPool2D::new((2, 2), Some((2, 2)), None, None, None, Some(true))
        .expect("maxpool construction");
    let input_data: Vec<f32> = (1..=9).map(|v| v as f32).collect();
    let input = make_tensor(input_data, &[1, 1, 3, 3]);
    let out = pool.forward(&input).expect("forward");
    assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 1, 2, 2]);
    let out_vec = out.tensor.to_vec().expect("out vec");
    // Boundary windows overhang the input; -inf fill means each takes the max
    // of its real elements only: [max(1,2,4,5), max(3,6), max(7,8), max(9)].
    assert_eq!(out_vec, vec![5.0, 6.0, 8.0, 9.0]);
}

#[test]
fn max_pool2d_dilation_real_values() {
    // kernel 2x2, dilation 2, stride 1, valid -> 2x2 output over a 4x4 input.
    let pool = PyMaxPool2D::new((2, 2), Some((1, 1)), None, Some((2, 2)), None, None)
        .expect("maxpool construction");
    let input_data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let input = make_tensor(input_data, &[1, 1, 4, 4]);
    let out = pool.forward(&input).expect("forward");
    assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 1, 2, 2]);
    let out_vec = out.tensor.to_vec().expect("out vec");
    // (0,0)=max(1,3,9,11)=11, (0,1)=max(2,4,10,12)=12,
    // (1,0)=max(5,7,13,15)=15, (1,1)=max(6,8,14,16)=16.
    assert_eq!(out_vec, vec![11.0, 12.0, 15.0, 16.0]);
}

#[test]
fn avg_pool2d_forward_real_values() {
    let pool = PyAvgPool2D::new((2, 2), Some((2, 2)), None, None, None, None)
        .expect("avgpool construction");
    let input_data: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let input = make_tensor(input_data, &[1, 1, 4, 4]);
    let out = pool.forward(&input).expect("forward");
    assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 1, 2, 2]);
    let out_vec = out.tensor.to_vec().expect("out vec");
    assert_eq!(out_vec, vec![2.5, 4.5, 10.5, 12.5]);
}

#[test]
fn avg_pool2d_explicit_padding_real_values() {
    // 2x2 input [[1,2],[3,4]], kernel 2, stride 1, padding 1. The core divides
    // every window by the full kernel area (count_include_pad=True default).
    let pool = PyAvgPool2D::new((2, 2), Some((1, 1)), Some((1, 1)), None, None, None)
        .expect("avgpool construction");
    let input = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    let out = pool.forward(&input).expect("forward");
    // padded 4x4, out = (4-2)/1 + 1 = 3.
    assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 1, 3, 3]);
    let out_vec = out.tensor.to_vec().expect("out vec");
    assert_eq!(
        out_vec,
        vec![0.25, 0.75, 0.5, 1.0, 2.5, 1.5, 0.75, 1.75, 1.0]
    );
}

#[test]
fn avg_pool2d_divisor_override_real_values() {
    // 4x4 input 1..16, kernel 2, stride 2, divisor_override=2.
    let pool = PyAvgPool2D::new((2, 2), Some((2, 2)), None, None, None, Some(2))
        .expect("avgpool construction");
    let input_data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let input = make_tensor(input_data, &[1, 1, 4, 4]);
    let out = pool.forward(&input).expect("forward");
    assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 1, 2, 2]);
    let out_vec = out.tensor.to_vec().expect("out vec");
    // sum(window)/2: (1+2+5+6)/2=7, (3+4+7+8)/2=11, (9+10+13+14)/2=23, (11+12+15+16)/2=27.
    assert_eq!(out_vec, vec![7.0, 11.0, 23.0, 27.0]);
}

#[test]
fn avg_pool2d_divisor_override_with_padding() {
    // Interaction case: divisor_override together with explicit padding. Every
    // window (including padding-touched boundary windows) is divided by 2.
    let pool = PyAvgPool2D::new((2, 2), Some((1, 1)), Some((1, 1)), None, None, Some(2))
        .expect("avgpool construction");
    let input = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    let out = pool.forward(&input).expect("forward");
    assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 1, 3, 3]);
    let out_vec = out.tensor.to_vec().expect("out vec");
    // window sums /2: 1/2, 3/2, 2/2, 4/2, 10/2, 6/2, 3/2, 7/2, 4/2.
    assert_eq!(out_vec, vec![0.5, 1.5, 1.0, 2.0, 5.0, 3.0, 1.5, 3.5, 2.0]);
}

#[test]
fn avg_pool2d_ceil_mode_all_ones_oracle() {
    // All-ones oracle: the average of ones is always exactly 1.0, no matter how
    // many real elements are averaged. A 5x5 all-ones input with kernel 2,
    // stride 2, ceil_mode=True yields a 3x3 output whose last row/col are
    // boundary/overhang windows. Every element MUST be exactly 1.0, proving the
    // ceil-overhang divisor exclusion.
    let pool = PyAvgPool2D::new((2, 2), Some((2, 2)), None, Some(true), None, None)
        .expect("avgpool construction");
    let input = make_tensor(vec![1.0; 25], &[1, 1, 5, 5]);
    let out = pool.forward(&input).expect("forward");
    assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 1, 3, 3]);
    let out_vec = out.tensor.to_vec().expect("out vec");
    for v in out_vec {
        assert!(
            (v - 1.0).abs() < 1e-6,
            "every all-ones average must equal 1.0, got {v}"
        );
    }
}

#[test]
fn conv1d_forward_real_values() {
    Python::initialize();
    Python::attach(|py| {
        reset_autograd_state();
        let conv = PyConv1D::new(py, 1, 1, 2, None, None, None, None, Some(false))
            .expect("conv1d construction");
        let weight = Tensor::from_vec(vec![1.0, 1.0], &[1, 1, 2]).expect("weight");
        conv.weight_param.borrow(py).set_data(weight).expect("set_data");
        let input = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let out = conv.forward(py, &input).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 1, 3]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        assert_eq!(out_vec, vec![3.0, 5.0, 7.0]);
    });
}

#[test]
fn conv1d_dilation_real_values() {
    Python::initialize();
    Python::attach(|py| {
        reset_autograd_state();
        // in=1, out=1, kernel 2, dilation 2, valid, all-ones weight.
        let conv = PyConv1D::new(py, 1, 1, 2, None, None, Some(2), None, Some(false))
            .expect("conv1d construction");
        let weight = Tensor::from_vec(vec![1.0, 1.0], &[1, 1, 2]).expect("weight");
        conv.weight_param.borrow(py).set_data(weight).expect("set_data");
        let input = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5]);
        let out = conv.forward(py, &input).expect("forward");
        // output_length = (5 - (2-1)*2 - 1)/1 + 1 = 3.
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 1, 3]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        // out[0]=in[0]+in[2]=1+3=4, out[1]=in[1]+in[3]=2+4=6, out[2]=in[2]+in[4]=3+5=8.
        assert_eq!(out_vec, vec![4.0, 6.0, 8.0]);
    });
}

#[test]
fn conv1d_groups_real_values() {
    Python::initialize();
    Python::attach(|py| {
        reset_autograd_state();
        // groups=2: output channel `oc` only sees its own group's input channel.
        let conv = PyConv1D::new(py, 2, 2, 1, None, None, None, Some(2), Some(false))
            .expect("conv1d construction");
        // Weight [out=2, in/groups=1, 1]: oc0 scales by 10, oc1 scales by 100.
        let weight = Tensor::from_vec(vec![10.0, 100.0], &[2, 1, 1]).expect("weight");
        conv.weight_param.borrow(py).set_data(weight).expect("set_data");
        // Input [1, 2, 3]: channel 0 = 2.0, channel 1 = 3.0.
        let input = make_tensor(vec![2.0, 2.0, 2.0, 3.0, 3.0, 3.0], &[1, 2, 3]);
        let out = conv.forward(py, &input).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 2, 3]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        // oc0 = 2*10 = 20 (would be 3*10 = 30 if groups were ignored); oc1 = 3*100 = 300.
        assert_eq!(out_vec, vec![20.0, 20.0, 20.0, 300.0, 300.0, 300.0]);
    });
}

#[test]
fn conv3d_forward_real_values() {
    Python::initialize();
    Python::attach(|py| {
        reset_autograd_state();
        // in=1, out=1, kernel (2,2,2), valid, all-ones weight.
        let conv = PyConv3D::new(py, 1, 1, (2, 2, 2), None, None, None, None, Some(false))
            .expect("conv3d construction");
        let weight = Tensor::from_vec(vec![1.0; 8], &[1, 1, 2, 2, 2]).expect("weight");
        conv.weight_param.borrow(py).set_data(weight).expect("set_data");
        // Input [1, 1, 3, 3, 3] = 27 elements, values 1..27.
        let input_data: Vec<f32> = (1..=27).map(|v| v as f32).collect();
        let input = make_tensor(input_data, &[1, 1, 3, 3, 3]);
        let out = conv.forward(py, &input).expect("forward");
        // out = (3-2)/1 + 1 = 2 per spatial axis.
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 1, 2, 2, 2]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        // Cross-check against the core op invoked directly.
        let direct = tenflowers_core::ops::conv3d(
            input.tensor.as_ref(),
            &Tensor::from_vec(vec![1.0; 8], &[1, 1, 2, 2, 2]).expect("weight2"),
            None,
            (1, 1, 1),
            "valid",
        )
        .expect("direct conv3d");
        let direct_vec = direct.to_vec().expect("direct vec");
        assert_eq!(out_vec, direct_vec);
    });
}

#[test]
fn conv3d_groups_real_values() {
    Python::initialize();
    Python::attach(|py| {
        reset_autograd_state();
        // groups=2: output channel `oc` only sees its own group's input channel.
        let conv = PyConv3D::new(py, 2, 2, (1, 1, 1), None, None, None, Some(2), Some(false))
            .expect("conv3d construction");
        // Weight [out=2, in/groups=1, 1, 1, 1]: oc0 scales by 10, oc1 scales by 100.
        let weight = Tensor::from_vec(vec![10.0, 100.0], &[2, 1, 1, 1, 1]).expect("weight");
        conv.weight_param.borrow(py).set_data(weight).expect("set_data");
        // Input [1, 2, 1, 1, 1]: channel 0 = 2.0, channel 1 = 3.0.
        let input = make_tensor(vec![2.0, 3.0], &[1, 2, 1, 1, 1]);
        let out = conv.forward(py, &input).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 2, 1, 1, 1]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        // oc0 = 2*10 = 20 (would be 3*10 = 30 if groups were ignored); oc1 = 3*100 = 300.
        assert_eq!(out_vec, vec![20.0, 300.0]);
    });
}

#[test]
fn conv3d_dilation_real_values() {
    Python::initialize();
    Python::attach(|py| {
        reset_autograd_state();
        // 1 in / 1 out, kernel (2,2,2), dilation (2,2,2), valid, all-ones weight.
        let conv = PyConv3D::new(
            py,
            1,
            1,
            (2, 2, 2),
            None,
            None,
            Some((2, 2, 2)),
            None,
            Some(false),
        )
        .expect("conv3d construction");
        let weight = Tensor::from_vec(vec![1.0; 8], &[1, 1, 2, 2, 2]).expect("weight");
        conv.weight_param.borrow(py).set_data(weight).expect("set_data");
        // 4x4x4 input, values 1..64.
        let input_data: Vec<f32> = (1..=64).map(|v| v as f32).collect();
        let input = make_tensor(input_data, &[1, 1, 4, 4, 4]);
        let out = conv.forward(py, &input).expect("forward");
        // effective kernel = 3, out = (4-3)/1 + 1 = 2 per spatial axis.
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 1, 2, 2, 2]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        assert!(out_vec.iter().any(|&v| v != 0.0));
    });
}

// -----------------------------------------------------------------
// Implicit-autograd integration tests: forward -> loss(sum) ->
// backward -> assert weight AND bias gradients populated and
// numerically correct (finite-difference checked against a small
// input), for Conv1D/Conv2D/Conv3D; forward -> loss(sum) -> backward
// -> assert gradient flows to a tracked predecessor, for
// MaxPool2D/AvgPool2D.
// -----------------------------------------------------------------

/// Central-difference numerical gradient of `f` with respect to each
/// element of `values`, perturbing one element at a time. Used to
/// independently verify the analytic gradients the implicit tape
/// produces for the tests below, exactly the way
/// `tenflowers_autograd::tape::helpers::compute_numerical_gradient`
/// verifies the tape's own analytic gradients internally.
fn numerical_gradient(values: &[f32], epsilon: f32, mut f: impl FnMut(&[f32]) -> f32) -> Vec<f32> {
    let mut grad = vec![0.0f32; values.len()];
    let mut perturbed = values.to_vec();
    for i in 0..values.len() {
        perturbed[i] = values[i] + epsilon;
        let plus = f(&perturbed);
        perturbed[i] = values[i] - epsilon;
        let minus = f(&perturbed);
        perturbed[i] = values[i];
        grad[i] = (plus - minus) / (2.0 * epsilon);
    }
    grad
}

/// `f(input, weight, bias) = sum(conv1d(input, weight, bias))` as a
/// plain, tape-free numerical oracle: builds fresh tensors from the
/// given flat data every call so `numerical_gradient`'s repeated calls
/// with perturbed values are fully independent of each other and of the
/// tape.
fn conv1d_sum_oracle<'a>(
    input_data: &'a [f32],
    input_shape: &'a [usize],
    weight_shape: &'a [usize],
    bias_len: usize,
    stride: usize,
) -> impl Fn(&[f32], &[f32], &[f32]) -> f32 + 'a {
    move |input_vals: &[f32], weight_vals: &[f32], bias_vals: &[f32]| {
        let _ = input_data; // captured shape only; values come from `input_vals`.
        let input = Tensor::from_vec(input_vals.to_vec(), input_shape).expect("input");
        let weight = Tensor::from_vec(weight_vals.to_vec(), weight_shape).expect("weight");
        let bias = Tensor::from_vec(bias_vals.to_vec(), &[bias_len]).expect("bias");
        let out = tenflowers_core::ops::conv1d(&input, &weight, Some(&bias), stride, "valid")
            .expect("conv1d");
        tenflowers_core::ops::sum(&out, None, false)
            .expect("sum")
            .to_vec()
            .expect("scalar readable")[0]
    }
}

#[test]
fn conv1d_backward_gradients_match_finite_difference() {
    Python::initialize();
    Python::attach(|py| {
        reset_autograd_state();
        let conv = PyConv1D::new(py, 1, 2, 2, None, None, None, None, Some(true))
            .expect("conv1d construction");
        let weight_data = vec![0.5, -0.3, 0.2, 0.7];
        let weight = Tensor::from_vec(weight_data.clone(), &[2, 1, 2]).expect("weight");
        conv.weight_param.borrow(py).set_data(weight).expect("set_data");
        let bias_data = vec![0.1, -0.2];
        let bias_param = conv
            .bias_param
            .as_ref()
            .expect("conv1d with bias=True must have a bias_param");
        bias_param
            .borrow(py)
            .set_data(Tensor::from_vec(bias_data.clone(), &[2]).expect("bias"))
            .expect("set_data");
        let weight_id = conv.weight_param.borrow(py).id();
        let bias_id = bias_param.borrow(py).id();

        let input_data = vec![1.0, 2.0, 3.0, -1.5];
        let input_shape = [1usize, 1, 4];
        let mut input = make_tensor(input_data.clone(), &input_shape);
        input.requires_grad = true;
        implicit_autograd::mark_leaf(&input);

        let out = conv.forward(py, &input).expect("forward");
        let summed = tenflowers_core::ops::sum(&out.tensor, None, false).expect("sum");
        let scalar = PyTensor {
            tensor: Arc::new(summed),
            requires_grad: true,
            is_pinned: false,
        };
        implicit_autograd::record_and_link_unary(
            crate::implicit_autograd::UnaryOpKind::Sum {
                axes: None,
                keepdims: false,
            },
            &out,
            &scalar,
        )
        .expect("recording sum must succeed");

        implicit_autograd::run_backward(&scalar).expect("backward must succeed");

        let grad_weight = implicit_autograd::get_grad_by_id(weight_id)
            .expect("weight gradient must be reachable")
            .tensor
            .to_vec()
            .expect("weight grad readable");
        let grad_bias = implicit_autograd::get_grad_by_id(bias_id)
            .expect("bias gradient must be reachable")
            .tensor
            .to_vec()
            .expect("bias grad readable");
        let grad_input = implicit_autograd::get_grad(&input)
            .expect("input gradient must be reachable")
            .tensor
            .to_vec()
            .expect("input grad readable");

        let oracle = conv1d_sum_oracle(&input_data, &input_shape, &[2, 1, 2], 2, 1);
        let epsilon = 1e-3;
        let numeric_weight = numerical_gradient(&weight_data, epsilon, |w| {
            oracle(&input_data, w, &bias_data)
        });
        let numeric_bias = numerical_gradient(&bias_data, epsilon, |b| {
            oracle(&input_data, &weight_data, b)
        });
        let numeric_input = numerical_gradient(&input_data, epsilon, |inp| {
            oracle(inp, &weight_data, &bias_data)
        });

        for (analytic, numeric) in grad_weight.iter().zip(numeric_weight.iter()) {
            assert!(
                (analytic - numeric).abs() < 1e-2,
                "conv1d weight grad mismatch: analytic={analytic}, numeric={numeric}"
            );
        }
        for (analytic, numeric) in grad_bias.iter().zip(numeric_bias.iter()) {
            assert!(
                (analytic - numeric).abs() < 1e-2,
                "conv1d bias grad mismatch: analytic={analytic}, numeric={numeric}"
            );
        }
        for (analytic, numeric) in grad_input.iter().zip(numeric_input.iter()) {
            assert!(
                (analytic - numeric).abs() < 1e-2,
                "conv1d input grad mismatch: analytic={analytic}, numeric={numeric}"
            );
        }
    });
}

#[test]
fn conv2d_backward_gradients_match_finite_difference() {
    Python::initialize();
    Python::attach(|py| {
        reset_autograd_state();
        let conv = PyConv2D::new(py, 1, 1, (2, 2), None, None, None, None, Some(true))
            .expect("conv2d construction");
        let weight_data = vec![0.4, -0.1, 0.3, 0.2];
        conv.weight_param
            .borrow(py)
            .set_data(Tensor::from_vec(weight_data.clone(), &[1, 1, 2, 2]).expect("weight"))
            .expect("set_data");
        let bias_data = vec![0.05];
        let bias_param = conv
            .bias_param
            .as_ref()
            .expect("conv2d with bias=True must have a bias_param");
        bias_param
            .borrow(py)
            .set_data(Tensor::from_vec(bias_data.clone(), &[1]).expect("bias"))
            .expect("set_data");
        let weight_id = conv.weight_param.borrow(py).id();
        let bias_id = bias_param.borrow(py).id();

        let input_data = vec![1.0, 0.5, -0.5, 2.0, 1.5, -1.0, 0.2, 0.8, -0.3];
        let input_shape = [1usize, 1, 3, 3];
        let mut input = make_tensor(input_data.clone(), &input_shape);
        input.requires_grad = true;
        implicit_autograd::mark_leaf(&input);

        let out = conv.forward(py, &input).expect("forward");
        let summed = tenflowers_core::ops::sum(&out.tensor, None, false).expect("sum");
        let scalar = PyTensor {
            tensor: Arc::new(summed),
            requires_grad: true,
            is_pinned: false,
        };
        implicit_autograd::record_and_link_unary(
            crate::implicit_autograd::UnaryOpKind::Sum {
                axes: None,
                keepdims: false,
            },
            &out,
            &scalar,
        )
        .expect("recording sum must succeed");

        implicit_autograd::run_backward(&scalar).expect("backward must succeed");

        let grad_weight = implicit_autograd::get_grad_by_id(weight_id)
            .expect("weight gradient must be reachable")
            .tensor
            .to_vec()
            .expect("weight grad readable");
        let grad_bias = implicit_autograd::get_grad_by_id(bias_id)
            .expect("bias gradient must be reachable")
            .tensor
            .to_vec()
            .expect("bias grad readable");

        let oracle = |w: &[f32], b: &[f32]| -> f32 {
            let input = Tensor::from_vec(input_data.clone(), &input_shape).expect("input");
            let weight = Tensor::from_vec(w.to_vec(), &[1, 1, 2, 2]).expect("weight");
            let bias = Tensor::from_vec(b.to_vec(), &[1]).expect("bias");
            let out = tenflowers_core::ops::conv2d(&input, &weight, Some(&bias), (1, 1), "valid")
                .expect("conv2d");
            tenflowers_core::ops::sum(&out, None, false)
                .expect("sum")
                .to_vec()
                .expect("scalar readable")[0]
        };
        let epsilon = 1e-3;
        let numeric_weight =
            numerical_gradient(&weight_data, epsilon, |w| oracle(w, &bias_data));
        let numeric_bias = numerical_gradient(&bias_data, epsilon, |b| oracle(&weight_data, b));

        for (analytic, numeric) in grad_weight.iter().zip(numeric_weight.iter()) {
            assert!(
                (analytic - numeric).abs() < 1e-2,
                "conv2d weight grad mismatch: analytic={analytic}, numeric={numeric}"
            );
        }
        for (analytic, numeric) in grad_bias.iter().zip(numeric_bias.iter()) {
            assert!(
                (analytic - numeric).abs() < 1e-2,
                "conv2d bias grad mismatch: analytic={analytic}, numeric={numeric}"
            );
        }
    });
}

#[test]
fn conv3d_backward_gradients_match_finite_difference() {
    Python::initialize();
    Python::attach(|py| {
        reset_autograd_state();
        let conv = PyConv3D::new(py, 1, 1, (2, 2, 2), None, None, None, None, Some(true))
            .expect("conv3d construction");
        let weight_data = vec![0.3, -0.2, 0.1, 0.4, -0.1, 0.2, 0.15, -0.05];
        conv.weight_param
            .borrow(py)
            .set_data(Tensor::from_vec(weight_data.clone(), &[1, 1, 2, 2, 2]).expect("weight"))
            .expect("set_data");
        let bias_data = vec![0.02];
        let bias_param = conv
            .bias_param
            .as_ref()
            .expect("conv3d with bias=True must have a bias_param");
        bias_param
            .borrow(py)
            .set_data(Tensor::from_vec(bias_data.clone(), &[1]).expect("bias"))
            .expect("set_data");
        let weight_id = conv.weight_param.borrow(py).id();
        let bias_id = bias_param.borrow(py).id();

        // 3x3x3 input keeps the finite-difference check small but still
        // real (27 elements).
        let input_data: Vec<f32> = (0..27).map(|v| (v as f32) * 0.1 - 1.0).collect();
        let input_shape = [1usize, 1, 3, 3, 3];
        let mut input = make_tensor(input_data.clone(), &input_shape);
        input.requires_grad = true;
        implicit_autograd::mark_leaf(&input);

        let out = conv.forward(py, &input).expect("forward");
        let summed = tenflowers_core::ops::sum(&out.tensor, None, false).expect("sum");
        let scalar = PyTensor {
            tensor: Arc::new(summed),
            requires_grad: true,
            is_pinned: false,
        };
        implicit_autograd::record_and_link_unary(
            crate::implicit_autograd::UnaryOpKind::Sum {
                axes: None,
                keepdims: false,
            },
            &out,
            &scalar,
        )
        .expect("recording sum must succeed");

        implicit_autograd::run_backward(&scalar).expect("backward must succeed");

        let grad_weight = implicit_autograd::get_grad_by_id(weight_id)
            .expect("weight gradient must be reachable")
            .tensor
            .to_vec()
            .expect("weight grad readable");
        let grad_bias = implicit_autograd::get_grad_by_id(bias_id)
            .expect("bias gradient must be reachable")
            .tensor
            .to_vec()
            .expect("bias grad readable");

        let oracle = |w: &[f32], b: &[f32]| -> f32 {
            let input = Tensor::from_vec(input_data.clone(), &input_shape).expect("input");
            let weight = Tensor::from_vec(w.to_vec(), &[1, 1, 2, 2, 2]).expect("weight");
            let bias = Tensor::from_vec(b.to_vec(), &[1]).expect("bias");
            let out =
                tenflowers_core::ops::conv3d(&input, &weight, Some(&bias), (1, 1, 1), "valid")
                    .expect("conv3d");
            tenflowers_core::ops::sum(&out, None, false)
                .expect("sum")
                .to_vec()
                .expect("scalar readable")[0]
        };
        let epsilon = 1e-3;
        let numeric_weight =
            numerical_gradient(&weight_data, epsilon, |w| oracle(w, &bias_data));
        let numeric_bias = numerical_gradient(&bias_data, epsilon, |b| oracle(&weight_data, b));

        for (analytic, numeric) in grad_weight.iter().zip(numeric_weight.iter()) {
            assert!(
                (analytic - numeric).abs() < 1e-2,
                "conv3d weight grad mismatch: analytic={analytic}, numeric={numeric}"
            );
        }
        for (analytic, numeric) in grad_bias.iter().zip(numeric_bias.iter()) {
            assert!(
                (analytic - numeric).abs() < 1e-2,
                "conv3d bias grad mismatch: analytic={analytic}, numeric={numeric}"
            );
        }
    });
}

#[test]
fn max_pool2d_backward_gradient_flows_to_predecessor() {
    Python::initialize();
    reset_autograd_state();
    // Chain MaxPool2D after a tracked input: input -> maxpool -> sum ->
    // backward, then confirm the INPUT's gradient (not maxpool's own,
    // since it has no parameters) is populated and correct. For max
    // pooling, d(sum(maxpool(x)))/dx is 1.0 at exactly the position that
    // won each window's max, and 0.0 everywhere else.
    let input_data: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let mut input = make_tensor(input_data, &[1, 1, 4, 4]);
    input.requires_grad = true;
    implicit_autograd::mark_leaf(&input);

    let pool = PyMaxPool2D::new((2, 2), Some((2, 2)), None, None, None, None)
        .expect("maxpool construction");
    let out = pool.forward(&input).expect("forward");
    assert_eq!(
        out.tensor.to_vec().expect("out vec"),
        vec![5.0, 7.0, 13.0, 15.0]
    );

    let summed = tenflowers_core::ops::sum(&out.tensor, None, false).expect("sum");
    let scalar = PyTensor {
        tensor: Arc::new(summed),
        requires_grad: true,
        is_pinned: false,
    };
    implicit_autograd::record_and_link_unary(
        crate::implicit_autograd::UnaryOpKind::Sum {
            axes: None,
            keepdims: false,
        },
        &out,
        &scalar,
    )
    .expect("recording sum must succeed");

    implicit_autograd::run_backward(&scalar).expect("backward must succeed");

    let grad_input = implicit_autograd::get_grad(&input)
        .expect("gradient must flow through MaxPool2D to its predecessor")
        .tensor
        .to_vec()
        .expect("input grad readable");
    // Input laid out row-major 4x4: [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]].
    // Each 2x2 window's max is its bottom-right element: 5, 7, 13, 15.
    let expected = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
    ];
    assert_eq!(
        grad_input, expected,
        "gradient must be 1.0 exactly at each window's argmax and 0.0 elsewhere"
    );
}

#[test]
fn avg_pool2d_backward_gradient_flows_to_predecessor() {
    Python::initialize();
    reset_autograd_state();
    // Chain AvgPool2D after a tracked input: input -> avgpool -> sum ->
    // backward, then confirm the INPUT's gradient is populated and
    // correct. For average pooling, d(sum(avgpool(x)))/dx is uniformly
    // 1/(kh*kw) everywhere (every input element contributes to exactly
    // one non-overlapping window here, since stride == kernel_size).
    let input_data: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let mut input = make_tensor(input_data, &[1, 1, 4, 4]);
    input.requires_grad = true;
    implicit_autograd::mark_leaf(&input);

    let pool = PyAvgPool2D::new((2, 2), Some((2, 2)), None, None, None, None)
        .expect("avgpool construction");
    let out = pool.forward(&input).expect("forward");
    assert_eq!(
        out.tensor.to_vec().expect("out vec"),
        vec![2.5, 4.5, 10.5, 12.5]
    );

    let summed = tenflowers_core::ops::sum(&out.tensor, None, false).expect("sum");
    let scalar = PyTensor {
        tensor: Arc::new(summed),
        requires_grad: true,
        is_pinned: false,
    };
    implicit_autograd::record_and_link_unary(
        crate::implicit_autograd::UnaryOpKind::Sum {
            axes: None,
            keepdims: false,
        },
        &out,
        &scalar,
    )
    .expect("recording sum must succeed");

    implicit_autograd::run_backward(&scalar).expect("backward must succeed");

    let grad_input = implicit_autograd::get_grad(&input)
        .expect("gradient must flow through AvgPool2D to its predecessor")
        .tensor
        .to_vec()
        .expect("input grad readable");
    let expected = [0.25f32; 16];
    for (analytic, exp) in grad_input.iter().zip(expected.iter()) {
        assert!(
            (analytic - exp).abs() < 1e-6,
            "gradient must be uniformly 1/(kh*kw)=0.25, got {analytic}"
        );
    }
}

#[test]
fn parameters_share_identity_with_forward_and_grad_is_reachable_through_them() {
    // Proves `.parameters()` returns handles sharing the EXACT identity
    // `forward()` reads (the core property `PyDense::parameters`'
    // struct-level doc establishes for this whole design): after a
    // backward pass, `.grad()` on a handle obtained from `parameters()`
    // (not from the internal `weight_param` field directly) must be
    // populated.
    Python::initialize();
    Python::attach(|py| {
        reset_autograd_state();
        let conv = PyConv2D::new(py, 1, 1, (2, 2), None, None, None, None, Some(true))
            .expect("conv2d construction");
        conv.weight_param
            .borrow(py)
            .set_data(Tensor::from_vec(vec![1.0; 4], &[1, 1, 2, 2]).expect("weight"))
            .expect("set_data");

        let params = conv.parameters(py);
        assert_eq!(params.len(), 2, "weight + bias with bias=True");
        let weight_handle_id = params[0].borrow(py).id();
        assert_eq!(
            weight_handle_id,
            conv.weight_param.borrow(py).id(),
            "parameters()'s first handle must share weight_param's exact id"
        );

        let input = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        let out = conv.forward(py, &input).expect("forward");
        let summed = tenflowers_core::ops::sum(&out.tensor, None, false).expect("sum");
        let scalar = PyTensor {
            tensor: Arc::new(summed),
            requires_grad: true,
            is_pinned: false,
        };
        implicit_autograd::record_and_link_unary(
            crate::implicit_autograd::UnaryOpKind::Sum {
                axes: None,
                keepdims: false,
            },
            &out,
            &scalar,
        )
        .expect("recording sum must succeed");
        implicit_autograd::run_backward(&scalar).expect("backward must succeed");

        // .grad() called on the handle FROM parameters(), not on
        // conv.weight_param directly - this is the property under test.
        let grad = params[0]
            .borrow(py)
            .grad()
            .expect("grad must be reachable via the parameters() handle");
        let grad_data = grad.tensor.to_vec().expect("grad data readable");
        assert!(
            grad_data.iter().any(|&v| v != 0.0),
            "gradient reached via parameters() must be real, non-zero data"
        );
    });
}

#[test]
fn set_data_through_parameters_handle_is_visible_to_next_forward() {
    // The other half of the identity-sharing guarantee: writing through a
    // handle returned by `parameters()` (as an optimizer's `.step()` would
    // via `.set_data()`) must be visible to this SAME layer's next
    // `forward()` call — not just readable via `.grad()` afterward. Mirrors
    // `PyDense::parameters`'s own documented guarantee.
    Python::initialize();
    Python::attach(|py| {
        reset_autograd_state();
        let conv = PyConv1D::new(py, 1, 1, 2, None, None, None, None, Some(false))
            .expect("conv1d construction");

        let params = conv.parameters(py);
        params[0]
            .borrow(py)
            .set_data(Tensor::from_vec(vec![2.0, 3.0], &[1, 1, 2]).expect("weight"))
            .expect("set_data via parameters() handle must succeed");

        let input = make_tensor(vec![1.0, 1.0, 1.0], &[1, 1, 3]);
        let out = conv.forward(py, &input).expect("forward");
        let out_vec = out.tensor.to_vec().expect("out vec");
        // weight [2, 3] applied to constant-1 input: each output position is
        // 2*1 + 3*1 = 5. If forward() were still reading a stale (zero)
        // weight, this would be all zeros instead.
        assert_eq!(
            out_vec,
            vec![5.0, 5.0],
            "forward() must observe the value written through parameters()'s handle"
        );
    });
}

// -----------------------------------------------------------------
// End-to-end system test: a tiny composed conv net (Conv2D -> ReLU ->
// MaxPool2D -> flatten -> Dense) trained with a real optimizer, proving
// gradients genuinely flow through EVERY layer type together — not just
// each layer type in isolation, which is all every test above (and
// `layers.rs`'s own Dense-only convergence tests) actually exercises.
// -----------------------------------------------------------------

/// Deterministic, small, non-zero, asymmetric weight seed: `sin` of a
/// scaled index keeps every value in `(-0.1, 0.1)` (small enough that the
/// net's initial forward pass is nowhere near saturated) while varying
/// smoothly and non-repetitively across indices (unlike e.g. a fixed
/// arithmetic ramp, this never produces two equal values by construction
/// for the small index ranges used below), with no RNG crate involved so
/// the whole test stays fully reproducible run-to-run.
fn deterministic_seed(i: usize) -> f32 {
    (i as f32 * 0.037).sin() * 0.1
}

/// Build the two fixed, hand-constructed 8x8 "images" used as this test's
/// entire (tiny, 2-sample) training set, batched into one `[2, 1, 8, 8]`
/// input tensor, plus their `[2, 1]` regression targets.
///
/// Pattern A (target `1.0`) is a checkerboard of alternating `1.0`/`0.2`;
/// pattern B (target `0.0`) is uniformly `0.1`. The two patterns are
/// trivially linearly separable by total intensity, so a tiny conv net
/// with a linear regression head can realistically drive the loss down
/// from this fixed starting point without needing a large step count.
fn checkerboard_vs_uniform_batch() -> (PyTensor, PyTensor) {
    let mut pattern_a = Vec::with_capacity(64);
    for r in 0..8usize {
        for c in 0..8usize {
            pattern_a.push(if (r + c) % 2 == 0 { 1.0 } else { 0.2 });
        }
    }
    let pattern_b = vec![0.1f32; 64];

    let mut batch_data = Vec::with_capacity(128);
    batch_data.extend_from_slice(&pattern_a);
    batch_data.extend_from_slice(&pattern_b);
    let x = make_tensor(batch_data, &[2, 1, 8, 8]);

    let y = make_tensor(vec![1.0, 0.0], &[2, 1]);
    (x, y)
}

/// Run one full forward pass through the composed conv net (Conv2D -> ReLU
/// -> MaxPool2D -> flatten -> Dense) for
/// `conv_net_end_to_end_training_drives_loss_down_through_every_layer`.
///
/// Kept as a standalone helper (rather than inlined in the training loop)
/// so the exact same forward path is used both inside the loop and for the
/// one-off before/after weight inspection at the end of that test.
fn conv_net_forward(
    py: Python<'_>,
    conv: &PyConv2D,
    pool: &PyMaxPool2D,
    dense: &PyDense,
    x: &PyTensor,
) -> PyResult<PyTensor> {
    let conv_out = conv.forward(py, x)?;
    let relu_out = crate::neural::functions::relu(&conv_out)?;
    let pooled = pool.forward(&relu_out)?;
    // Pooled shape is [2, 4, 3, 3] for the fixed [2, 1, 8, 8] input and
    // (3,3)-kernel/4-out-channel Conv2D + (2,2) MaxPool2D configuration
    // this test uses throughout (8 -> conv valid 3x3 -> 6 -> pool 2x2 -> 3),
    // computed by hand once and passed as an exact literal target shape,
    // exactly as the task spec calls for.
    let flattened = crate::tensor_ops::reshape(&pooled, vec![2, 4 * 3 * 3])?;
    dense.forward(&flattened)
}

/// Build the Python-side wrapper object `collect_parameters` needs:
/// `PySGD`/`PyAdam::step()`/`zero_grad()` both call `model.parameters()`
/// with zero Python-level arguments (see `optimizer_bridge.rs`'s
/// `collect_parameters`), but this test's composed model is two separate
/// layer objects (`PyConv2D` + `PyDense`, `PyMaxPool2D` has no parameters
/// to expose), not a single `#[pyclass]` exposing one combined
/// `.parameters()`. This mirrors `optimizers.rs`'s own `FakeModel` test
/// fixture (`make_single_param_model`) at the module level: a tiny
/// pure-Python class, built once via `PyModule::from_code`, that holds
/// stable references to the SAME `Py<PyConv2D>`/`Py<PyDense>` objects the
/// caller forwards through directly, and whose `.parameters()` simply
/// concatenates both layers' own `.parameters()` results — so parameter
/// identity is shared between what this test trains through and what the
/// optimizer sees, exactly like every other layer's `parameters()`
/// identity-sharing guarantee documented elsewhere in this module.
fn make_conv_dense_model<'py>(
    py: Python<'py>,
    conv: Py<PyConv2D>,
    dense: Py<PyDense>,
) -> PyResult<Bound<'py, PyAny>> {
    let code = std::ffi::CString::new(
        r#"
class ConvDenseModel:
    def __init__(self, conv, dense):
        self._conv = conv
        self._dense = dense
    def parameters(self):
        return list(self._conv.parameters()) + list(self._dense.parameters())
"#,
    )
    .expect("test source has no NUL bytes");
    let module = PyModule::from_code(
        py,
        code.as_c_str(),
        c"conv_dense_model_test_module.py",
        c"conv_dense_model_test_module",
    )?;
    let make_model = module.getattr("ConvDenseModel")?;
    make_model.call1((conv, dense))
}

#[test]
fn conv_net_end_to_end_training_drives_loss_down_through_every_layer() {
    Python::initialize();
    Python::attach(|py| {
        reset_autograd_state();

        // Conv2D: in=1, out=4, kernel 3x3, all other args left at their
        // tape-recordable defaults (stride=1, padding=0, dilation=1,
        // groups=1 — see this module's `PyConv2D::forward` doc for exactly
        // why any of those being non-default would silently break
        // `.backward()` through this layer), with bias.
        let conv = PyConv2D::new(py, 1, 4, (3, 3), None, None, None, None, Some(true))
            .expect("conv2d construction");
        // Weight [out=4, in=1, 3, 3] = 36 elements; bias [4]. Both are
        // zero-initialised by `PyConv2D::new` (see `zero_parameter_handle`'s
        // doc) and MUST be overwritten to a non-zero, asymmetric starting
        // point before training - an all-zero Conv2D weight is a symmetric
        // fixed point every output channel starts identical from, which can
        // produce systematically degenerate gradients.
        let conv_weight_data: Vec<f32> = (0..36).map(deterministic_seed).collect();
        conv.weight_param
            .borrow(py)
            .set_data(Tensor::from_vec(conv_weight_data.clone(), &[4, 1, 3, 3]).expect("conv weight"))
            .expect("seed conv weight");
        let conv_bias_data: Vec<f32> = (0..4).map(|i| deterministic_seed(i + 100)).collect();
        let conv_bias_param = conv
            .bias_param
            .as_ref()
            .expect("conv2d with bias=True must have a bias_param");
        conv_bias_param
            .borrow(py)
            .set_data(Tensor::from_vec(conv_bias_data, &[4]).expect("conv bias"))
            .expect("seed conv bias");

        // MaxPool2D: 2x2 kernel, all other args left at their
        // tape-recordable defaults (stride defaults to kernel_size,
        // padding=0, dilation=1, no ceil-mode overhang, return_indices=False).
        let pool = PyMaxPool2D::new((2, 2), None, None, None, None, None)
            .expect("maxpool2d construction");

        // Dense head: flattened pooled features (4 channels * 3 * 3 spatial
        // = 36) -> 1 regression output, with bias, no activation (MSE is
        // applied directly to the raw output).
        let dense = PyDense::new(4 * 3 * 3, 1, Some(true), None).expect("dense construction");
        // `PyDense::new` already Xavier-initialises its weight (unlike
        // Conv2D - see `PyDense::new`'s own doc), but this test re-seeds it
        // too anyway for full determinism (Xavier init here draws from an
        // unseeded RNG - see `optimizers.rs`'s `fresh_zero_init_dense` doc
        // for the same rationale applied to a bare PyDense), matching this
        // test's "deterministic only" requirement across every parameter.
        let dense_params = dense.parameters();
        assert_eq!(dense_params.len(), 2, "PyDense(36, 1, bias=true) must expose [weight, bias]");
        let dense_weight_data: Vec<f32> = (0..36).map(|i| deterministic_seed(i + 200)).collect();
        dense_params[0]
            .borrow(py)
            .set_data(Tensor::from_vec(dense_weight_data.clone(), &[36, 1]).expect("dense weight"))
            .expect("seed dense weight");
        let dense_bias_data = vec![deterministic_seed(300)];
        dense_params[1]
            .borrow(py)
            .set_data(Tensor::from_vec(dense_bias_data.clone(), &[1]).expect("dense bias"))
            .expect("seed dense bias");

        let (x, y) = checkerboard_vs_uniform_batch();

        // Wrap the SAME Py<PyConv2D>/Py<PyDense> objects this test forwards
        // through directly so parameter identity is shared with what the
        // optimizer sees (see `make_conv_dense_model`'s doc).
        let py_conv = Py::new(py, conv).expect("Py::new(conv)");
        let py_dense = Py::new(py, dense).expect("Py::new(dense)");

        let num_steps = 200;
        let lr = 0.05_f64;

        // --- Initial forward pass (before any training) for the loss-drop
        // assertion and the before/after weight-change assertion below. ---
        let initial_weight_snapshot = {
            let conv_ref = py_conv.borrow(py);
            let dense_ref = py_dense.borrow(py);
            let out = conv_net_forward(py, &conv_ref, &pool, &dense_ref, &x)
                .expect("initial forward must succeed");
            let loss = crate::neural::losses::mse_loss(&out, &y, None)
                .expect("initial mse_loss must succeed");
            let initial_loss = loss.tensor.to_vec().expect("initial loss readable")[0];

            let conv_weight_before = conv_ref
                .weight_param
                .borrow(py)
                .to_tensor()
                .expect("conv weight snapshot")
                .tensor
                .to_vec()
                .expect("conv weight vec");
            let dense_weight_before = dense_ref.parameters()[0]
                .borrow(py)
                .to_tensor()
                .expect("dense weight snapshot")
                .tensor
                .to_vec()
                .expect("dense weight vec");
            (initial_loss, conv_weight_before, dense_weight_before)
        };
        let (initial_loss, conv_weight_before, dense_weight_before) = initial_weight_snapshot;
        assert!(
            initial_loss.is_finite() && initial_loss > 0.0,
            "initial loss must be a real, positive number, got {initial_loss}"
        );

        // --- Real training loop: forward -> mse_loss -> run_backward ->
        // SGD::step -> SGD::zero_grad, all through the real production
        // path (collect_parameters via the Python wrapper's .parameters()). ---
        let mut sgd = PySGD::new(Some(lr));
        let mut final_loss = initial_loss;
        for _ in 0..num_steps {
            reset_autograd_state();
            let out = {
                let conv_ref = py_conv.borrow(py);
                let dense_ref = py_dense.borrow(py);
                conv_net_forward(py, &conv_ref, &pool, &dense_ref, &x)
                    .expect("training forward must succeed")
            };
            let loss = crate::neural::losses::mse_loss(&out, &y, None)
                .expect("training mse_loss must succeed");
            final_loss = loss.tensor.to_vec().expect("training loss readable")[0];

            crate::implicit_autograd::run_backward(&loss).expect("run_backward must succeed");

            let model_for_step = make_conv_dense_model(py, py_conv.clone_ref(py), py_dense.clone_ref(py))
                .expect("model wrapper for step()");
            sgd.step(model_for_step).expect("sgd.step must succeed");

            let model_for_zero_grad =
                make_conv_dense_model(py, py_conv.clone_ref(py), py_dense.clone_ref(py))
                    .expect("model wrapper for zero_grad()");
            sgd.zero_grad(model_for_zero_grad).expect("sgd.zero_grad must succeed");
        }

        // --- Assertion (a): loss substantially lower than initial loss. ---
        assert!(
            final_loss < initial_loss * 0.5,
            "final loss ({final_loss}) must be substantially lower than initial loss \
             ({initial_loss}) after {num_steps} real SGD steps at lr={lr}"
        );

        // --- Assertion (b): gradients genuinely flowed through EVERY layer
        // - both the Conv2D weight AND the Dense weight must have actually
        // changed from their seeded initial values. This is exactly the
        // system-level gap no per-layer test suite catches: it is entirely
        // possible for only the outermost Dense head's gradient to flow
        // while the Conv2D layer underneath is silently frozen. ---
        let conv_weight_after = py_conv
            .borrow(py)
            .weight_param
            .borrow(py)
            .to_tensor()
            .expect("conv weight snapshot after training")
            .tensor
            .to_vec()
            .expect("conv weight vec after training");
        let dense_weight_after = py_dense.borrow(py).parameters()[0]
            .borrow(py)
            .to_tensor()
            .expect("dense weight snapshot after training")
            .tensor
            .to_vec()
            .expect("dense weight vec after training");

        assert_ne!(
            conv_weight_after, conv_weight_before,
            "Conv2D weight must have changed after training - if it did not, gradients \
             silently failed to flow through the Conv2D layer"
        );
        assert_ne!(
            dense_weight_after, dense_weight_before,
            "Dense weight must have changed after training - if it did not, gradients \
             silently failed to flow through the Dense layer"
        );

        // ---------------------------------------------------------------
        // Negative control (proves the two assertions above are real, not
        // vacuously true): rerun the ENTIRE experiment from the same fixed
        // seeded starting point, but with the SGD learning rate neutered to
        // 0.0 - a legitimate negative control that exercises the exact same
        // production `.step()` code path with the update mathematically
        // disabled (`w -= 0.0 * grad == w`), without touching optimizers.rs
        // at all. If the assertions above were vacuous (e.g. a bug made
        // `final_loss`/the weight snapshots always differ from initial
        // regardless of training), this block would incorrectly pass too;
        // it does not, which is exactly what demonstrates the real
        // assertions above would catch a broken/no-op optimizer.
        // ---------------------------------------------------------------
        reset_autograd_state();
        let neg_conv = PyConv2D::new(py, 1, 4, (3, 3), None, None, None, None, Some(true))
            .expect("negative-control conv2d construction");
        neg_conv
            .weight_param
            .borrow(py)
            .set_data(Tensor::from_vec(conv_weight_data, &[4, 1, 3, 3]).expect("neg conv weight"))
            .expect("seed negative-control conv weight");
        let neg_conv_bias_data: Vec<f32> = (0..4).map(|i| deterministic_seed(i + 100)).collect();
        neg_conv
            .bias_param
            .as_ref()
            .expect("negative-control conv2d with bias=True must have a bias_param")
            .borrow(py)
            .set_data(Tensor::from_vec(neg_conv_bias_data, &[4]).expect("neg conv bias"))
            .expect("seed negative-control conv bias");
        let neg_pool = PyMaxPool2D::new((2, 2), None, None, None, None, None)
            .expect("negative-control maxpool2d construction");
        let neg_dense = PyDense::new(4 * 3 * 3, 1, Some(true), None)
            .expect("negative-control dense construction");
        let neg_dense_params = neg_dense.parameters();
        neg_dense_params[0]
            .borrow(py)
            .set_data(Tensor::from_vec(dense_weight_data, &[36, 1]).expect("neg dense weight"))
            .expect("seed negative-control dense weight");
        neg_dense_params[1]
            .borrow(py)
            .set_data(Tensor::from_vec(dense_bias_data, &[1]).expect("neg dense bias"))
            .expect("seed negative-control dense bias");

        let neg_py_conv = Py::new(py, neg_conv).expect("Py::new(neg_conv)");
        let neg_py_dense = Py::new(py, neg_dense).expect("Py::new(neg_dense)");

        let neg_initial = {
            let conv_ref = neg_py_conv.borrow(py);
            let dense_ref = neg_py_dense.borrow(py);
            let out = conv_net_forward(py, &conv_ref, &neg_pool, &dense_ref, &x)
                .expect("negative-control initial forward must succeed");
            crate::neural::losses::mse_loss(&out, &y, None)
                .expect("negative-control initial mse_loss must succeed")
                .tensor
                .to_vec()
                .expect("negative-control initial loss readable")[0]
        };
        let neg_conv_weight_before = neg_py_conv
            .borrow(py)
            .weight_param
            .borrow(py)
            .to_tensor()
            .expect("neg conv weight snapshot")
            .tensor
            .to_vec()
            .expect("neg conv weight vec");
        let neg_dense_weight_before = neg_py_dense.borrow(py).parameters()[0]
            .borrow(py)
            .to_tensor()
            .expect("neg dense weight snapshot")
            .tensor
            .to_vec()
            .expect("neg dense weight vec");

        // The neutered optimizer: lr=0.0 is the deliberate negative control.
        let mut neutered_sgd = PySGD::new(Some(0.0_f64));
        let mut neg_final_loss = neg_initial;
        for _ in 0..num_steps {
            reset_autograd_state();
            let out = {
                let conv_ref = neg_py_conv.borrow(py);
                let dense_ref = neg_py_dense.borrow(py);
                conv_net_forward(py, &conv_ref, &neg_pool, &dense_ref, &x)
                    .expect("negative-control training forward must succeed")
            };
            let loss = crate::neural::losses::mse_loss(&out, &y, None)
                .expect("negative-control training mse_loss must succeed");
            neg_final_loss = loss.tensor.to_vec().expect("negative-control training loss readable")[0];

            crate::implicit_autograd::run_backward(&loss)
                .expect("negative-control run_backward must succeed");

            let model_for_step =
                make_conv_dense_model(py, neg_py_conv.clone_ref(py), neg_py_dense.clone_ref(py))
                    .expect("negative-control model wrapper for step()");
            neutered_sgd.step(model_for_step).expect("neutered sgd.step must succeed");

            let model_for_zero_grad =
                make_conv_dense_model(py, neg_py_conv.clone_ref(py), neg_py_dense.clone_ref(py))
                    .expect("negative-control model wrapper for zero_grad()");
            neutered_sgd
                .zero_grad(model_for_zero_grad)
                .expect("neutered sgd.zero_grad must succeed");
        }

        // With lr=0.0 the update is mathematically neutered (w -= 0*grad ==
        // w unchanged), so NEITHER of this test's two real assertions can
        // hold here: loss must NOT have dropped, and neither weight may have
        // changed. This proves the real (lr=0.05) assertions above are
        // genuinely exercising the update, not vacuously true.
        assert!(
            neg_final_loss >= neg_initial * 0.5,
            "negative control failed to prove itself: with lr=0.0 the loss must NOT drop \
             below half its initial value (initial={neg_initial}, final={neg_final_loss}) - \
             if it did, the real assertion above would not actually be verifying training"
        );
        let neg_conv_weight_after = neg_py_conv
            .borrow(py)
            .weight_param
            .borrow(py)
            .to_tensor()
            .expect("neg conv weight snapshot after")
            .tensor
            .to_vec()
            .expect("neg conv weight vec after");
        let neg_dense_weight_after = neg_py_dense.borrow(py).parameters()[0]
            .borrow(py)
            .to_tensor()
            .expect("neg dense weight snapshot after")
            .tensor
            .to_vec()
            .expect("neg dense weight vec after");
        assert_eq!(
            neg_conv_weight_after, neg_conv_weight_before,
            "negative control failed to prove itself: with lr=0.0 the Conv2D weight must NOT \
             change"
        );
        assert_eq!(
            neg_dense_weight_after, neg_dense_weight_before,
            "negative control failed to prove itself: with lr=0.0 the Dense weight must NOT \
             change"
        );
    });
}
