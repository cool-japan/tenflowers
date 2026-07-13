//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use crate::neural::layers::PyParameter;
use pyo3::prelude::*;
use tenflowers_core::Tensor;

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    use crate::implicit_autograd::{
        mark_leaf_param, record_and_link_binary, record_and_link_unary, run_backward, BinaryOpKind,
        UnaryOpKind,
    };
    use crate::tensor_ops::PyTensor;
    use std::ffi::CString;
    use std::sync::Arc as StdArc;
    /// Build a single-parameter "model" exposing `.parameters() -> [PyParameter]`,
    /// wrapping one scalar-valued parameter starting at `initial_value`.
    fn make_single_param_model(
        py: Python<'_>,
        initial_value: f32,
    ) -> (Bound<'_, PyAny>, Py<PyParameter>) {
        let code = CString::new(
            r#"
class OneParamModel:
    def __init__(self, param):
        self._param = param
    def parameters(self):
        return [self._param]
"#,
        )
        .expect("test source has no NUL bytes");
        let module = PyModule::from_code(
            py,
            code.as_c_str(),
            c"extended_optimizers_test_module.py",
            c"extended_optimizers_test_module",
        )
        .expect("test module should load");
        let class = module
            .getattr("OneParamModel")
            .expect("OneParamModel should be defined");
        let tensor = Tensor::from_vec(vec![initial_value], &[1])
            .expect("scalar tensor construction should succeed");
        let param = Py::new(
            py,
            PyParameter::new(
                PyTensor {
                    tensor: StdArc::new(tensor),
                    requires_grad: true,
                    is_pinned: false,
                },
                Some(true),
            ),
        )
        .expect("PyParameter should construct");
        let model = class
            .call1((param.clone_ref(py),))
            .expect("OneParamModel(...) should construct");
        (model, param)
    }
    /// Give `param` an exact, arbitrary gradient value by running a real
    /// (tiny) backward pass through the implicit-autograd tape: `L =
    /// sum(param * g)` has `dL/d(param) = g` everywhere, for any constant
    /// `g`, so this synthesizes precisely the gradient a test wants without
    /// needing any direct "poke the gradient store" backdoor (none exists —
    /// confirmed by reading `implicit_autograd/mod.rs` in full: `GRAD_STORE` is
    /// only ever written by `run_backward` itself).
    fn set_grad_via_backward(py: Python<'_>, param: &Py<PyParameter>, grad_value: f32) {
        let param_ref = param.borrow(py);
        let snapshot = param_ref.to_tensor().expect("to_tensor should succeed");
        mark_leaf_param(&snapshot, param_ref.id());
        drop(param_ref);
        let g_tensor = Tensor::from_vec(vec![grad_value], &[1])
            .expect("gradient-constant tensor construction should succeed");
        let g_py = PyTensor {
            tensor: StdArc::new(g_tensor),
            requires_grad: false,
            is_pinned: false,
        };
        let raw_mul =
            tenflowers_core::ops::mul(&snapshot.tensor, &g_py.tensor).expect("mul should succeed");
        let mul_result = PyTensor {
            tensor: StdArc::new(raw_mul),
            requires_grad: true,
            is_pinned: false,
        };
        record_and_link_binary(BinaryOpKind::Mul, &snapshot, &g_py, &mul_result)
            .expect("recording mul should succeed");
        let raw_sum = mul_result
            .tensor
            .sum(None, false)
            .expect("sum should succeed");
        let sum_result = PyTensor {
            tensor: StdArc::new(raw_sum),
            requires_grad: true,
            is_pinned: false,
        };
        record_and_link_unary(
            UnaryOpKind::Sum {
                axes: None,
                keepdims: false,
            },
            &mul_result,
            &sum_result,
        )
        .expect("recording sum should succeed");
        run_backward(&sum_result).expect("backward should succeed");
    }
    fn read_value(py: Python<'_>, param: &Py<PyParameter>) -> f32 {
        param
            .borrow(py)
            .to_tensor()
            .expect("to_tensor should succeed")
            .tensor
            .to_vec()
            .expect("to_vec should succeed")[0]
    }
    /// Drive `f(w) = (w - target)^2` (gradient `2*(w-target)`) for `steps`
    /// iterations of the given optimizer step closure, recording the
    /// parameter's value after every step.
    fn run_quadratic_descent(
        py: Python<'_>,
        param: &Py<PyParameter>,
        target: f32,
        steps: usize,
        mut step_fn: impl FnMut(Python<'_>),
    ) -> Vec<f32> {
        let mut trace = Vec::with_capacity(steps);
        for _ in 0..steps {
            let w = read_value(py, param);
            let grad = 2.0 * (w - target);
            set_grad_via_backward(py, param, grad);
            step_fn(py);
            trace.push(read_value(py, param));
        }
        trace
    }
    #[test]
    fn adabelief_moves_parameter_toward_target() {
        Python::initialize();
        Python::attach(|py| {
            let target = 5.0_f32;
            let (model, param) = make_single_param_model(py, 0.0);
            let mut opt = PyAdaBelief::with_betas(0.1, 0.9, 0.999);
            let trace = run_quadratic_descent(py, &param, target, 50, |py| {
                opt.step(model.clone())
                    .unwrap_or_else(|e| panic!("AdaBelief step should succeed: {}", e));
                let _ = py;
            });
            let start_distance = (0.0_f32 - target).abs();
            let end_distance = (trace[trace.len() - 1] - target).abs();
            assert!(
                end_distance < start_distance,
                "AdaBelief should move the parameter toward the target: trace={:?}",
                trace
            );
            let early = (trace[4] - target).abs();
            let late = (trace[trace.len() - 1] - target).abs();
            assert!(
                late < early,
                "AdaBelief should keep converging over time: early={}, late={}, trace={:?}",
                early,
                late,
                trace
            );
            println!("AdaBelief trace (target={}): {:?}", target, trace);
        });
    }
    #[test]
    fn adabelief_amsgrad_variant_also_converges() {
        Python::initialize();
        Python::attach(|py| {
            let target = -3.0_f32;
            let (model, param) = make_single_param_model(py, 2.0);
            let mut opt = PyAdaBelief::with_amsgrad(0.1, true);
            assert!(opt.amsgrad);
            let trace = run_quadratic_descent(py, &param, target, 50, |py| {
                opt.step(model.clone())
                    .unwrap_or_else(|e| panic!("AdaBelief(amsgrad) step should succeed: {}", e));
                let _ = py;
            });
            let start_distance = (2.0_f32 - target).abs();
            let end_distance = (trace[trace.len() - 1] - target).abs();
            assert!(
                end_distance < start_distance * 0.5,
                "AdaBelief with amsgrad should substantially converge: trace={:?}",
                trace
            );
        });
    }
    #[test]
    fn radam_moves_parameter_toward_target() {
        Python::initialize();
        Python::attach(|py| {
            let target = 4.0_f32;
            let (model, param) = make_single_param_model(py, 0.0);
            let mut opt = PyRAdam::with_betas(0.5, 0.9, 0.999);
            let trace = run_quadratic_descent(py, &param, target, 200, |py| {
                opt.step(model.clone())
                    .unwrap_or_else(|e| panic!("RAdam step should succeed: {}", e));
                let _ = py;
            });
            let start_distance = (0.0_f32 - target).abs();
            let end_distance = (trace[trace.len() - 1] - target).abs();
            assert!(
                end_distance < start_distance,
                "RAdam should move the parameter toward the target: trace={:?}",
                trace
            );
            let mid_distance = (trace[9] - target).abs();
            let late_distance = (trace[trace.len() - 1] - target).abs();
            assert!(
                late_distance < mid_distance,
                "RAdam should keep converging past the rectification warm-up: \
                 mid={}, late={}, trace={:?}",
                mid_distance,
                late_distance,
                trace
            );
            println!("RAdam trace (target={}): {:?}", target, trace);
        });
    }
    #[test]
    fn radam_early_steps_use_unrectified_sgd_like_fallback() {
        Python::initialize();
        Python::attach(|py| {
            let (model, param) = make_single_param_model(py, 1.0);
            let mut opt = PyRAdam::with_betas(0.1, 0.9, 0.999);
            set_grad_via_backward(py, &param, 2.0);
            opt.step(model.clone())
                .unwrap_or_else(|e| panic!("RAdam first step should succeed: {}", e));
            let w = read_value(py, &param);
            assert!(w.is_finite(), "RAdam first-step result must be finite");
            assert!(
                w < 1.0,
                "RAdam's first-step unrectified fallback should still move opposite the \
                 positive gradient (w started at 1.0, grad=2.0): got {}",
                w
            );
        });
    }
    #[test]
    fn nadam_moves_parameter_toward_target() {
        Python::initialize();
        Python::attach(|py| {
            let target = 6.0_f32;
            let (model, param) = make_single_param_model(py, 0.0);
            let mut opt = PyNadam::with_betas(0.1, 0.9, 0.999);
            let trace = run_quadratic_descent(py, &param, target, 50, |py| {
                opt.step(model.clone())
                    .unwrap_or_else(|e| panic!("Nadam step should succeed: {}", e));
                let _ = py;
            });
            let start_distance = (0.0_f32 - target).abs();
            let end_distance = (trace[trace.len() - 1] - target).abs();
            assert!(
                end_distance < start_distance,
                "Nadam should move the parameter toward the target: trace={:?}",
                trace
            );
            let early = (trace[4] - target).abs();
            let late = (trace[trace.len() - 1] - target).abs();
            assert!(
                late < early,
                "Nadam should keep converging over time: early={}, late={}, trace={:?}",
                early,
                late,
                trace
            );
            println!("Nadam trace (target={}): {:?}", target, trace);
        });
    }
    #[test]
    fn adagrad_moves_parameter_toward_target() {
        Python::initialize();
        Python::attach(|py| {
            let target = 7.0_f32;
            let (model, param) = make_single_param_model(py, 0.0);
            let mut opt = PyAdaGrad::new(Some(1.0));
            let trace = run_quadratic_descent(py, &param, target, 100, |py| {
                opt.step(model.clone())
                    .unwrap_or_else(|e| panic!("AdaGrad step should succeed: {}", e));
                let _ = py;
            });
            let start_distance = (0.0_f32 - target).abs();
            let end_distance = (trace[trace.len() - 1] - target).abs();
            assert!(
                end_distance < start_distance,
                "AdaGrad should move the parameter toward the target: trace={:?}",
                trace
            );
            let early = (trace[9] - target).abs();
            let late = (trace[trace.len() - 1] - target).abs();
            assert!(
                late < early,
                "AdaGrad should keep converging over time: early={}, late={}, trace={:?}",
                early,
                late,
                trace
            );
            println!("AdaGrad trace (target={}): {:?}", target, trace);
        });
    }
    #[test]
    fn adagrad_lr_decay_reduces_effective_learning_rate_over_time() {
        Python::initialize();
        Python::attach(|py| {
            let opt = PyAdaGrad::with_lr_decay(1.0, 1.0);
            assert_eq!(opt.get_learning_rate(), 1.0);
            let target = 1.0_f32;
            let (model, param) = make_single_param_model(py, 0.0);
            let mut opt = PyAdaGrad::with_lr_decay(1.0, 1.0);
            let trace = run_quadratic_descent(py, &param, target, 20, |py| {
                opt.step(model.clone())
                    .unwrap_or_else(|e| panic!("AdaGrad(lr_decay) step should succeed: {}", e));
                let _ = py;
            });
            assert!(opt.get_learning_rate() < 1.0);
            let end_distance = (trace[trace.len() - 1] - target).abs();
            assert!(
                end_distance < (0.0_f32 - target).abs(),
                "AdaGrad with lr_decay should still make progress toward the target: {:?}",
                trace
            );
        });
    }
    #[test]
    fn adadelta_moves_parameter_toward_target() {
        Python::initialize();
        Python::attach(|py| {
            let target = 3.0_f32;
            let (model, param) = make_single_param_model(py, 0.0);
            let mut opt = PyAdaDelta::new(Some(0.9));
            let trace = run_quadratic_descent(py, &param, target, 300, |py| {
                opt.step(model.clone())
                    .unwrap_or_else(|e| panic!("AdaDelta step should succeed: {}", e));
                let _ = py;
            });
            let start_distance = (0.0_f32 - target).abs();
            let end_distance = (trace[trace.len() - 1] - target).abs();
            assert!(
                end_distance < start_distance,
                "AdaDelta should move the parameter toward the target: trace={:?}",
                trace
            );
            let mid_distance = (trace[49] - target).abs();
            let late_distance = (trace[trace.len() - 1] - target).abs();
            assert!(
                late_distance < mid_distance,
                "AdaDelta should keep converging over time: mid={}, late={}, trace={:?}",
                mid_distance,
                late_distance,
                trace
            );
            println!("AdaDelta trace (target={}): {:?}", target, trace);
        });
    }
    #[test]
    fn parameters_without_gradients_are_skipped_not_errored() {
        Python::initialize();
        Python::attach(|py| {
            let (model, param) = make_single_param_model(py, 42.0);
            let mut opt = PyAdaBelief::new(Some(0.1));
            opt.step(model.clone())
                .expect("step() must not error when a parameter has no gradient");
            assert_eq!(
                read_value(py, &param),
                42.0,
                "a parameter with no gradient must be left unchanged"
            );
            let mut opt = PyRAdam::new(Some(0.1));
            opt.step(model.clone())
                .expect("RAdam step() must not error when a parameter has no gradient");
            let mut opt = PyNadam::new(Some(0.1));
            opt.step(model.clone())
                .expect("Nadam step() must not error when a parameter has no gradient");
            let mut opt = PyAdaGrad::new(Some(0.1));
            opt.step(model.clone())
                .expect("AdaGrad step() must not error when a parameter has no gradient");
            let mut opt = PyAdaDelta::new(Some(0.9));
            opt.step(model.clone())
                .expect("AdaDelta step() must not error when a parameter has no gradient");
            assert_eq!(
                read_value(py, &param),
                42.0,
                "value must remain unchanged after every optimizer's no-gradient step"
            );
        });
    }
    #[test]
    fn zero_grad_clears_gradient_so_next_grad_call_errors() {
        Python::initialize();
        Python::attach(|py| {
            let (model, param) = make_single_param_model(py, 1.0);
            set_grad_via_backward(py, &param, 9.0);
            assert!(param.borrow(py).grad().is_ok());
            let opt = PyAdaBelief::new(Some(0.1));
            opt.zero_grad(model.clone())
                .expect("zero_grad should succeed");
            assert!(
                param.borrow(py).grad().is_err(),
                "grad() should error after zero_grad() clears the gradient store entry"
            );
        });
    }
    #[test]
    fn state_persists_across_multiple_step_calls() {
        Python::initialize();
        Python::attach(|py| {
            let (model, param) = make_single_param_model(py, 0.0);
            let mut opt = PyAdaGrad::new(Some(1.0));
            set_grad_via_backward(py, &param, 1.0);
            opt.step(model.clone()).expect("step 1");
            let w1 = read_value(py, &param);
            let step1 = -w1;
            set_grad_via_backward(py, &param, 1.0);
            opt.step(model.clone()).expect("step 2");
            let w2 = read_value(py, &param);
            let step2 = w1 - w2;
            assert!(
                step2 < step1,
                "AdaGrad's step size must shrink as its accumulator persists and grows \
                 across calls: step1={}, step2={} (w1={}, w2={})",
                step1,
                step2,
                w1,
                w2
            );
        });
    }
    /// Fixed, hand-verified 4-row dataset for `y = 2*x0 + 3*x1 + 1`:
    /// `(0,0)->1`, `(1,0)->3`, `(0,1)->4`, `(1,1)->6`.
    fn fixed_linear_regression_dataset(py: Python<'_>) -> (PyTensor, PyTensor) {
        let _ = py;
        let x_tensor =
            Tensor::<f32>::from_vec(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[4, 2])
                .expect("test: x tensor construction should succeed");
        let x = PyTensor {
            tensor: StdArc::new(x_tensor),
            requires_grad: false,
            is_pinned: false,
        };
        let y_tensor = Tensor::<f32>::from_vec(vec![1.0, 3.0, 4.0, 6.0], &[4, 1])
            .expect("test: y tensor construction should succeed");
        let y = PyTensor {
            tensor: StdArc::new(y_tensor),
            requires_grad: false,
            is_pinned: false,
        };
        (x, y)
    }
    /// Assert the loss trend is unambiguously, non-trivially decreasing:
    /// the mean of the first 10 recorded losses must be at least
    /// `1/margin_divisor` times larger than the mean of the last 10. Not a
    /// strict every-step-decreases check (RAdam's rectification warm-up
    /// and AdaGrad/AdaBelief's early behaviour are known from this same
    /// module's other tests to be non-monotonic early on), but a
    /// comfortably-robust "genuinely converged" margin, not a knife-edge
    /// "didn't diverge" one.
    fn assert_loss_trend_decreasing(trace: &[f32], margin_divisor: f32, optimizer_name: &str) {
        assert!(
            trace.len() >= 20,
            "{optimizer_name}: trace too short to assess an early/late trend: {:?}",
            trace
        );
        let early_mean = trace[0..10].iter().sum::<f32>() / 10.0;
        let late_slice = &trace[trace.len() - 10..];
        let late_mean = late_slice.iter().sum::<f32>() / 10.0;
        assert!(
            late_mean < early_mean / margin_divisor,
            "{optimizer_name}: loss should trend down over training (early_mean={}, \
             late_mean={}, required margin_divisor={}): trace={:?}",
            early_mean,
            late_mean,
            margin_divisor,
            trace
        );
    }
    #[test]
    fn adabelief_step_drives_real_pydense_loss_down_over_many_steps() {
        Python::initialize();
        Python::attach(|py| {
            let dense = crate::neural::layers::PyDense::new(2, 1, Some(true), None)
                .expect("test: PyDense::new should construct with activation=None");
            let py_dense =
                Py::new(py, dense).expect("test: PyDense should construct as a Py<PyDense>");
            let (x, y) = fixed_linear_regression_dataset(py);
            let mut opt = PyAdaBelief::with_betas(0.1, 0.9, 0.999);
            let mut trace: Vec<f32> = Vec::with_capacity(150);
            for _ in 0..150 {
                let output = py_dense
                    .borrow(py)
                    .forward(&x)
                    .expect("test: forward pass should succeed");
                let loss = crate::neural::losses::mse_loss(&output, &y, None)
                    .expect("test: mse_loss should succeed");
                crate::implicit_autograd::run_backward(&loss)
                    .expect("test: backward pass should succeed");
                let loss_value = loss
                    .tensor
                    .to_vec()
                    .expect("test: loss tensor should be readable")[0];
                trace.push(loss_value);
                let model_any: Bound<'_, PyAny> = py_dense.bind(py).clone().into_any();
                opt.step(model_any)
                    .expect("test: AdaBelief step over a real PyDense must succeed");
                let model_any2: Bound<'_, PyAny> = py_dense.bind(py).clone().into_any();
                opt.zero_grad(model_any2)
                    .expect("test: AdaBelief zero_grad must succeed");
            }
            assert_loss_trend_decreasing(&trace, 2.0, "AdaBelief");
        });
    }
    #[test]
    fn radam_step_drives_real_pydense_loss_down_over_many_steps() {
        Python::initialize();
        Python::attach(|py| {
            let dense = crate::neural::layers::PyDense::new(2, 1, Some(true), None)
                .expect("test: PyDense::new should construct with activation=None");
            let py_dense =
                Py::new(py, dense).expect("test: PyDense should construct as a Py<PyDense>");
            let (x, y) = fixed_linear_regression_dataset(py);
            let mut opt = PyRAdam::with_betas(0.05, 0.9, 0.999);
            let mut trace: Vec<f32> = Vec::with_capacity(400);
            for _ in 0..400 {
                let output = py_dense
                    .borrow(py)
                    .forward(&x)
                    .expect("test: forward pass should succeed");
                let loss = crate::neural::losses::mse_loss(&output, &y, None)
                    .expect("test: mse_loss should succeed");
                crate::implicit_autograd::run_backward(&loss)
                    .expect("test: backward pass should succeed");
                let loss_value = loss
                    .tensor
                    .to_vec()
                    .expect("test: loss tensor should be readable")[0];
                trace.push(loss_value);
                let model_any: Bound<'_, PyAny> = py_dense.bind(py).clone().into_any();
                opt.step(model_any)
                    .expect("test: RAdam step over a real PyDense must succeed");
                let model_any2: Bound<'_, PyAny> = py_dense.bind(py).clone().into_any();
                opt.zero_grad(model_any2)
                    .expect("test: RAdam zero_grad must succeed");
            }
            assert_loss_trend_decreasing(&trace, 2.0, "RAdam");
        });
    }
    #[test]
    fn nadam_step_drives_real_pydense_loss_down_over_many_steps() {
        Python::initialize();
        Python::attach(|py| {
            let dense = crate::neural::layers::PyDense::new(2, 1, Some(true), None)
                .expect("test: PyDense::new should construct with activation=None");
            let py_dense =
                Py::new(py, dense).expect("test: PyDense should construct as a Py<PyDense>");
            let (x, y) = fixed_linear_regression_dataset(py);
            let mut opt = PyNadam::with_betas(0.05, 0.9, 0.999);
            let mut trace: Vec<f32> = Vec::with_capacity(150);
            for _ in 0..150 {
                let output = py_dense
                    .borrow(py)
                    .forward(&x)
                    .expect("test: forward pass should succeed");
                let loss = crate::neural::losses::mse_loss(&output, &y, None)
                    .expect("test: mse_loss should succeed");
                crate::implicit_autograd::run_backward(&loss)
                    .expect("test: backward pass should succeed");
                let loss_value = loss
                    .tensor
                    .to_vec()
                    .expect("test: loss tensor should be readable")[0];
                trace.push(loss_value);
                let model_any: Bound<'_, PyAny> = py_dense.bind(py).clone().into_any();
                opt.step(model_any)
                    .expect("test: Nadam step over a real PyDense must succeed");
                let model_any2: Bound<'_, PyAny> = py_dense.bind(py).clone().into_any();
                opt.zero_grad(model_any2)
                    .expect("test: Nadam zero_grad must succeed");
            }
            assert_loss_trend_decreasing(&trace, 2.0, "Nadam");
        });
    }
    #[test]
    fn adagrad_step_drives_real_pydense_loss_down_over_many_steps() {
        Python::initialize();
        Python::attach(|py| {
            let dense = crate::neural::layers::PyDense::new(2, 1, Some(true), None)
                .expect("test: PyDense::new should construct with activation=None");
            let py_dense =
                Py::new(py, dense).expect("test: PyDense should construct as a Py<PyDense>");
            let (x, y) = fixed_linear_regression_dataset(py);
            let mut opt = PyAdaGrad::new(Some(1.0));
            let mut trace: Vec<f32> = Vec::with_capacity(300);
            for _ in 0..300 {
                let output = py_dense
                    .borrow(py)
                    .forward(&x)
                    .expect("test: forward pass should succeed");
                let loss = crate::neural::losses::mse_loss(&output, &y, None)
                    .expect("test: mse_loss should succeed");
                crate::implicit_autograd::run_backward(&loss)
                    .expect("test: backward pass should succeed");
                let loss_value = loss
                    .tensor
                    .to_vec()
                    .expect("test: loss tensor should be readable")[0];
                trace.push(loss_value);
                let model_any: Bound<'_, PyAny> = py_dense.bind(py).clone().into_any();
                opt.step(model_any)
                    .expect("test: AdaGrad step over a real PyDense must succeed");
                let model_any2: Bound<'_, PyAny> = py_dense.bind(py).clone().into_any();
                opt.zero_grad(model_any2)
                    .expect("test: AdaGrad zero_grad must succeed");
            }
            assert_loss_trend_decreasing(&trace, 2.0, "AdaGrad");
        });
    }
    #[test]
    fn adadelta_step_drives_real_pydense_loss_down_over_many_steps() {
        Python::initialize();
        Python::attach(|py| {
            let dense = crate::neural::layers::PyDense::new(2, 1, Some(true), None)
                .expect("test: PyDense::new should construct with activation=None");
            let py_dense =
                Py::new(py, dense).expect("test: PyDense should construct as a Py<PyDense>");
            let (x, y) = fixed_linear_regression_dataset(py);
            let mut opt = PyAdaDelta::new(Some(0.9));
            let mut trace: Vec<f32> = Vec::with_capacity(600);
            for _ in 0..600 {
                let output = py_dense
                    .borrow(py)
                    .forward(&x)
                    .expect("test: forward pass should succeed");
                let loss = crate::neural::losses::mse_loss(&output, &y, None)
                    .expect("test: mse_loss should succeed");
                crate::implicit_autograd::run_backward(&loss)
                    .expect("test: backward pass should succeed");
                let loss_value = loss
                    .tensor
                    .to_vec()
                    .expect("test: loss tensor should be readable")[0];
                trace.push(loss_value);
                let model_any: Bound<'_, PyAny> = py_dense.bind(py).clone().into_any();
                opt.step(model_any)
                    .expect("test: AdaDelta step over a real PyDense must succeed");
                let model_any2: Bound<'_, PyAny> = py_dense.bind(py).clone().into_any();
                opt.zero_grad(model_any2)
                    .expect("test: AdaDelta zero_grad must succeed");
            }
            assert_loss_trend_decreasing(&trace, 1.5, "AdaDelta");
        });
    }
    /// End-to-end multi-layer proof: a 3-layer `PySequential`
    /// (`Dense(2,8,relu) -> Dense(8,8,relu) -> Dense(8,1,None)`) trained
    /// with real `PySGD` against the same fixed linear-regression target,
    /// asserting (a) the loss trends down substantially and (b) every one
    /// of the 6 underlying parameters (weight+bias for each of the 3
    /// layers) individually changed value — the core "did gradient
    /// silently stop flowing partway through the composed model" check.
    #[test]
    fn mlp_end_to_end_training_drives_loss_down_through_every_layer() {
        Python::initialize();
        Python::attach(|py| {
            let mut seq = crate::neural::layers::PySequential::new();
            seq.add(
                crate::neural::layers::PyDense::new(2, 8, Some(true), Some("relu".to_string()))
                    .expect("test: layer 1 PyDense::new should succeed"),
            );
            seq.add(
                crate::neural::layers::PyDense::new(8, 8, Some(true), Some("relu".to_string()))
                    .expect("test: layer 2 PyDense::new should succeed"),
            );
            seq.add(
                crate::neural::layers::PyDense::new(8, 1, Some(true), None)
                    .expect("test: layer 3 PyDense::new should succeed"),
            );
            let py_seq = Py::new(py, seq)
                .expect("test: PySequential should construct as a Py<PySequential>");
            let params_before = py_seq.borrow(py).parameters();
            assert_eq!(
                params_before.len(),
                6,
                "3 Dense layers with bias each must expose exactly 6 parameters \
                 (weight+bias per layer), got {}",
                params_before.len()
            );
            let before: Vec<Vec<f32>> = params_before
                .iter()
                .map(|p| {
                    p.borrow(py)
                        .to_tensor()
                        .expect("test: to_tensor should succeed on a pre-training parameter")
                        .tensor
                        .to_vec()
                        .expect("test: to_vec should succeed on a pre-training parameter")
                })
                .collect();
            let (x, y) = fixed_linear_regression_dataset(py);
            let lr: f64 = 0.05;
            let mut sgd = crate::neural::optimizers::PySGD::new(Some(lr));
            let mut trace: Vec<f32> = Vec::with_capacity(400);
            for _ in 0..400 {
                let output = py_seq
                    .borrow(py)
                    .forward(&x)
                    .expect("test: forward pass should succeed");
                let loss = crate::neural::losses::mse_loss(&output, &y, None)
                    .expect("test: mse_loss should succeed");
                crate::implicit_autograd::run_backward(&loss)
                    .expect("test: backward pass should succeed");
                let loss_value = loss
                    .tensor
                    .to_vec()
                    .expect("test: loss tensor should be readable")[0];
                trace.push(loss_value);
                let model_any: Bound<'_, PyAny> = py_seq.bind(py).clone().into_any();
                sgd.step(model_any)
                    .expect("test: SGD step over a real PySequential must succeed");
                let model_any2: Bound<'_, PyAny> = py_seq.bind(py).clone().into_any();
                sgd.zero_grad(model_any2)
                    .expect("test: SGD zero_grad must succeed");
            }
            let initial_loss = trace[0];
            let final_loss = trace[trace.len() - 1];
            assert!(
                final_loss < initial_loss * 0.5,
                "MLP training should substantially reduce the loss: initial={}, final={}, \
                 trace={:?}",
                initial_loss,
                final_loss,
                trace
            );
            let params_after = py_seq.borrow(py).parameters();
            assert_eq!(params_after.len(), 6);
            let after: Vec<Vec<f32>> = params_after
                .iter()
                .map(|p| {
                    p.borrow(py)
                        .to_tensor()
                        .expect("test: to_tensor should succeed on a post-training parameter")
                        .tensor
                        .to_vec()
                        .expect("test: to_vec should succeed on a post-training parameter")
                })
                .collect();
            let param_labels = [
                "layer1.weight",
                "layer1.bias",
                "layer2.weight",
                "layer2.bias",
                "layer3.weight",
                "layer3.bias",
            ];
            for (idx, label) in param_labels.iter().enumerate() {
                let before_vals = &before[idx];
                let after_vals = &after[idx];
                let max_abs_diff = before_vals
                    .iter()
                    .zip(after_vals.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_abs_diff > 1e-4,
                    "parameter index {idx} ({label}) did not change nontrivially during \
                     training: max_abs_diff={max_abs_diff}, before={:?}, after={:?}",
                    before_vals,
                    after_vals
                );
            }
        });
    }
}
