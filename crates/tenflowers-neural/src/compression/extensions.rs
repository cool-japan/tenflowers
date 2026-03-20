use scirs2_core::random::{rngs::StdRng, Rng};
use scirs2_core::RngExt;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::VecDeque;
use std::fs;
use tenflowers_core::{Result, TensorError};

#[derive(Debug, Clone)]
pub struct ThroughputReport {
    pub n_requests: usize,
    pub mean_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub requests_per_sec: f64,
}

#[derive(Debug, Clone)]
pub struct ThroughputMonitor {
    latencies: Vec<f64>,
}

impl ThroughputMonitor {
    pub fn new() -> Self {
        Self {
            latencies: Vec::new(),
        }
    }

    pub fn record(&mut self, latency_ms: f64) {
        self.latencies.push(latency_ms);
    }

    pub fn report(&self) -> ThroughputReport {
        if self.latencies.is_empty() {
            return ThroughputReport {
                n_requests: 0,
                mean_latency_ms: 0.0,
                p50_latency_ms: 0.0,
                p95_latency_ms: 0.0,
                p99_latency_ms: 0.0,
                requests_per_sec: 0.0,
            };
        }

        let n = self.latencies.len();
        let mean = self.latencies.iter().sum::<f64>() / n as f64;

        let mut sorted = self.latencies.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        let percentile = |p: f64| -> f64 {
            let idx = ((p / 100.0) * (n as f64 - 1.0)).round() as usize;
            sorted[idx.min(n - 1)]
        };

        let total_time_s = self.latencies.iter().sum::<f64>() / 1000.0;
        let rps = if total_time_s > 1e-6 {
            n as f64 / total_time_s
        } else {
            0.0
        };

        ThroughputReport {
            n_requests: n,
            mean_latency_ms: mean,
            p50_latency_ms: percentile(50.0),
            p95_latency_ms: percentile(95.0),
            p99_latency_ms: percentile(99.0),
            requests_per_sec: rps,
        }
    }
}

impl Default for ThroughputMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct CachePage {
    pub page_id: usize,
    pub keys: Vec<Vec<f64>>,
    pub values: Vec<Vec<f64>>,
    pub used_slots: usize,
}

impl CachePage {
    fn new(page_id: usize, page_size: usize, head_dim: usize) -> Self {
        Self {
            page_id,
            keys: vec![vec![0.0; head_dim]; page_size],
            values: vec![vec![0.0; head_dim]; page_size],
            used_slots: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CacheManager {
    pub page_size: usize,
    pub head_dim: usize,
    pub max_pages: usize,
    pages: Vec<Option<CachePage>>,
    free_list: Vec<usize>,
    next_page_id: usize,
}

impl CacheManager {
    pub fn new(page_size: usize, head_dim: usize, max_pages: usize) -> Self {
        let free_list: Vec<usize> = (0..max_pages).collect();
        let pages = vec![None; max_pages];
        Self {
            page_size,
            head_dim,
            max_pages,
            pages,
            free_list,
            next_page_id: 0,
        }
    }

    pub fn allocate_page(&mut self) -> Result<usize> {
        match self.free_list.pop() {
            Some(slot) => {
                let id = self.next_page_id;
                self.next_page_id += 1;
                self.pages[slot] = Some(CachePage::new(id, self.page_size, self.head_dim));
                Ok(slot)
            }
            None => Err(TensorError::invalid_argument(
                "KV-cache is full: no free pages".to_string(),
            )),
        }
    }

    pub fn free_page(&mut self, slot: usize) -> Result<()> {
        if slot >= self.max_pages {
            return Err(TensorError::invalid_argument(
                "invalid page slot index".to_string(),
            ));
        }
        self.pages[slot] = None;
        self.free_list.push(slot);
        Ok(())
    }

    pub fn write_kv(&mut self, slot: usize, key: Vec<f64>, value: Vec<f64>) -> Result<()> {
        let page = self.pages[slot].as_mut().ok_or_else(|| {
            TensorError::invalid_argument("page slot is not allocated".to_string())
        })?;
        if page.used_slots >= self.page_size {
            return Err(TensorError::invalid_argument("page is full".to_string()));
        }
        page.keys[page.used_slots] = key;
        page.values[page.used_slots] = value;
        page.used_slots += 1;
        Ok(())
    }

    pub fn free_pages(&self) -> usize {
        self.free_list.len()
    }
}

#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub input: Vec<f64>,
    pub priority: i32,
    pub sequence: u64,
}

impl PartialEq for InferenceRequest {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.sequence == other.sequence
    }
}
impl Eq for InferenceRequest {}
impl PartialOrd for InferenceRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for InferenceRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

#[derive(Debug, Clone)]
pub struct RequestScheduler {
    heap: BinaryHeap<InferenceRequest>,
    sequence_counter: u64,
}

impl RequestScheduler {
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            sequence_counter: 0,
        }
    }

    pub fn submit(&mut self, input: Vec<f64>, priority: i32) {
        let seq = self.sequence_counter;
        self.sequence_counter += 1;
        self.heap.push(InferenceRequest {
            input,
            priority,
            sequence: seq,
        });
    }

    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<InferenceRequest> {
        self.heap.pop()
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

impl Default for RequestScheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct LayerProfile {
    pub layer_idx: usize,
    pub input_size: usize,
    pub output_size: usize,
    pub flops: u64,
    pub latency_us: f64,
}

#[derive(Debug, Clone)]
pub struct ModelProfiler {
    pub us_per_flop: f64,
}

impl ModelProfiler {
    pub fn new(us_per_flop: f64) -> Self {
        Self { us_per_flop }
    }

    pub fn profile_forward(&self, layer_sizes: &[(usize, usize)]) -> Vec<LayerProfile> {
        layer_sizes
            .iter()
            .enumerate()
            .map(|(idx, &(in_sz, out_sz))| {
                let flops = 2 * in_sz as u64 * out_sz as u64;
                let latency_us = flops as f64 * self.us_per_flop;
                LayerProfile {
                    layer_idx: idx,
                    input_size: in_sz,
                    output_size: out_sz,
                    flops,
                    latency_us,
                }
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct ExportLayer {
    pub layer_type: String,
    pub weight: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct TorchScriptExporter;

impl TorchScriptExporter {
    fn fmt_vec(v: &[f64]) -> String {
        let inner = v
            .iter()
            .map(|x| format!("{x:.6}"))
            .collect::<Vec<_>>()
            .join(",");
        format!("[{inner}]")
    }

    fn fmt_mat(m: &[Vec<f64>]) -> String {
        let rows = m
            .iter()
            .map(|r| Self::fmt_vec(r))
            .collect::<Vec<_>>()
            .join(",");
        format!("[{rows}]")
    }

    pub fn export(layers: &[ExportLayer], path: &str) -> Result<()> {
        let mut json = String::from("{\n  \"layers\": [\n");
        for (i, layer) in layers.iter().enumerate() {
            json.push_str(&format!(
                "    {{\"type\":\"{}\",\"weight\":{},\"bias\":{}}}",
                layer.layer_type,
                Self::fmt_mat(&layer.weight),
                Self::fmt_vec(&layer.bias),
            ));
            if i + 1 < layers.len() {
                json.push(',');
            }
            json.push('\n');
        }
        json.push_str("  ]\n}\n");
        fs::write(path, json)
            .map_err(|e| TensorError::invalid_argument(format!("failed to write export file: {e}")))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExtOnnxOpType {
    /// Standard linear / fully-connected layer.
    Linear,
    /// 1-D convolutional layer.
    Conv1d,
    /// Gated Recurrent Unit.
    Gru,
    /// Batch Normalisation.
    BatchNorm,
    /// ReLU activation.
    Relu,
    /// Softmax activation.
    Softmax,
    /// Custom / unrecognised op.
    Custom(String),
}

#[derive(Debug, Clone)]
pub struct ExtOnnxNode {
    pub op_type: ExtOnnxOpType,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
pub struct OnnxExporter {
    nodes: Vec<ExtOnnxNode>,
}

impl OnnxExporter {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn add_conv1d(&mut self, input: &str, output: &str, kernel_size: usize, stride: usize) {
        self.nodes.push(ExtOnnxNode {
            op_type: ExtOnnxOpType::Conv1d,
            inputs: vec![input.to_string()],
            outputs: vec![output.to_string()],
            attributes: vec![
                ("kernel_size".into(), kernel_size.to_string()),
                ("stride".into(), stride.to_string()),
            ],
        });
    }

    pub fn add_gru(&mut self, input: &str, output: &str, hidden_size: usize) {
        self.nodes.push(ExtOnnxNode {
            op_type: ExtOnnxOpType::Gru,
            inputs: vec![input.to_string()],
            outputs: vec![output.to_string()],
            attributes: vec![("hidden_size".into(), hidden_size.to_string())],
        });
    }

    pub fn add_batch_norm(&mut self, input: &str, output: &str, num_features: usize) {
        self.nodes.push(ExtOnnxNode {
            op_type: ExtOnnxOpType::BatchNorm,
            inputs: vec![input.to_string()],
            outputs: vec![output.to_string()],
            attributes: vec![("num_features".into(), num_features.to_string())],
        });
    }

    pub fn nodes(&self) -> &[ExtOnnxNode] {
        &self.nodes
    }

    pub fn export_json(&self, path: &str) -> Result<()> {
        let mut json = String::from("{\"nodes\":[\n");
        for (i, node) in self.nodes.iter().enumerate() {
            let op = match &node.op_type {
                ExtOnnxOpType::Linear => "Linear",
                ExtOnnxOpType::Conv1d => "Conv1d",
                ExtOnnxOpType::Gru => "GRU",
                ExtOnnxOpType::BatchNorm => "BatchNormalization",
                ExtOnnxOpType::Relu => "Relu",
                ExtOnnxOpType::Softmax => "Softmax",
                ExtOnnxOpType::Custom(s) => s.as_str(),
            };
            json.push_str(&format!(
                "  {{\"op\":\"{op}\",\"inputs\":{:?},\"outputs\":{:?}}}",
                node.inputs, node.outputs
            ));
            if i + 1 < self.nodes.len() {
                json.push(',');
            }
            json.push('\n');
        }
        json.push_str("]}\n");
        fs::write(path, json)
            .map_err(|e| TensorError::invalid_argument(format!("failed to write ONNX JSON: {e}")))
    }
}

impl Default for OnnxExporter {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct CoreMlLayerSpec {
    pub name: String,
    pub layer_kind: String,
    pub input_channels: usize,
    pub output_channels: usize,
    pub has_bias: bool,
}

#[derive(Debug, Clone)]
pub struct CoreMlSpec {
    pub spec_version: u32,
    pub layers: Vec<CoreMlLayerSpec>,
    pub input_name: String,
    pub output_name: String,
}

#[derive(Debug, Clone)]
pub struct CoreMlConverter;

impl CoreMlConverter {
    pub fn convert(layer_info: &[(String, String, usize, usize, bool)]) -> CoreMlSpec {
        let layers = layer_info
            .iter()
            .map(|(name, kind, in_c, out_c, bias)| CoreMlLayerSpec {
                name: name.clone(),
                layer_kind: kind.clone(),
                input_channels: *in_c,
                output_channels: *out_c,
                has_bias: *bias,
            })
            .collect();

        CoreMlSpec {
            spec_version: 5,
            layers,
            input_name: "input".to_string(),
            output_name: "output".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TfliteOpCode {
    FullyConnected = 9,
    Conv2d = 3,
    DepthwiseConv2d = 4,
    Relu = 19,
    Softmax = 25,
    Reshape = 22,
    Lstm = 35,
    Add = 0,
    Mul = 18,
}

#[derive(Debug, Clone)]
pub struct TfliteQuantParams {
    pub scale: f32,
    pub zero_point: i32,
    pub quantized_dimension: i32,
}

#[derive(Debug, Clone)]
pub struct TfliteOp {
    pub op_code: TfliteOpCode,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
    pub quant_params: Option<TfliteQuantParams>,
}

#[derive(Debug, Clone)]
pub struct TfliteModel {
    pub version: u32,
    pub ops: Vec<TfliteOp>,
    pub tensor_shapes: Vec<Vec<usize>>,
}

#[derive(Debug, Clone)]
pub struct TfliteConverter;

impl TfliteConverter {
    pub fn convert(
        ops: &[(TfliteOpCode, usize, usize)],
        tensor_shapes: Vec<Vec<usize>>,
    ) -> TfliteModel {
        let tflite_ops = ops
            .iter()
            .map(|&(code, in_idx, out_idx)| TfliteOp {
                op_code: code,
                inputs: vec![in_idx],
                outputs: vec![out_idx],
                quant_params: None,
            })
            .collect();
        TfliteModel {
            version: 3,
            ops: tflite_ops,
            tensor_shapes,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BenchmarkStats {
    pub n_iter: usize,
    pub mean_us: f64,
    pub std_us: f64,
    pub min_us: f64,
    pub max_us: f64,
    pub median_us: f64,
}

#[derive(Debug, Clone)]
pub struct BenchmarkRunner;

impl BenchmarkRunner {
    /// Run `n_iter` iterations over an input of `input_size` random values.
    pub fn run(input_size: usize, n_iter: usize, rng: &mut StdRng) -> BenchmarkStats {
        if n_iter == 0 || input_size == 0 {
            return BenchmarkStats {
                n_iter,
                mean_us: 0.0,
                std_us: 0.0,
                min_us: 0.0,
                max_us: 0.0,
                median_us: 0.0,
            };
        }

        let weight: Vec<Vec<f64>> = (0..input_size)
            .map(|_| {
                (0..input_size)
                    .map(|_| rng.random::<f64>() * 2.0 - 1.0)
                    .collect()
            })
            .collect();

        let mut latencies = Vec::with_capacity(n_iter);

        for _ in 0..n_iter {
            let input: Vec<f64> = (0..input_size).map(|_| rng.random::<f64>()).collect();

            let mut output = vec![0.0f64; input_size];
            for (i, row) in weight.iter().enumerate() {
                for (j, &w) in row.iter().enumerate() {
                    output[i] += w * input[j];
                }
            }

            let flops = 2 * input_size * input_size;
            let jitter = 1.0 + (rng.random::<f64>() - 0.5) * 0.1;
            let lat_us = flops as f64 * 1e-3 * jitter; // 1 ns/flop → µs
            latencies.push(lat_us);
        }

        let mean = latencies.iter().sum::<f64>() / n_iter as f64;
        let variance = latencies.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_iter as f64;
        let std = variance.sqrt();
        let min = latencies.iter().cloned().fold(f64::MAX, f64::min);
        let max = latencies.iter().cloned().fold(f64::MIN, f64::max);

        let mut sorted = latencies.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let median = sorted[n_iter / 2];

        BenchmarkStats {
            n_iter,
            mean_us: mean,
            std_us: std,
            min_us: min,
            max_us: max,
            median_us: median,
        }
    }
}

