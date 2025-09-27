//! Feature engineering transformations
//!
//! This module provides feature engineering transformations commonly used
//! for data preprocessing and feature extraction in machine learning.

use crate::transforms::Transform;
use scirs2_core::random::Rng;
use std::collections::HashMap;
use std::marker::PhantomData;
use tenflowers_core::{Result, Tensor, TensorError};

/// Polynomial features transformation
/// Generates polynomial and interaction features up to a given degree
pub struct PolynomialFeatures<T> {
    degree: usize,
    include_bias: bool,
    interaction_only: bool,
    _phantom: PhantomData<T>,
}

impl<T> PolynomialFeatures<T>
where
    T: Clone
        + Default
        + num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    pub fn new(degree: usize) -> Self {
        Self {
            degree: degree.max(1),
            include_bias: true,
            interaction_only: false,
            _phantom: PhantomData,
        }
    }

    pub fn with_bias(mut self, include_bias: bool) -> Self {
        self.include_bias = include_bias;
        self
    }

    pub fn interaction_only(mut self) -> Self {
        self.interaction_only = true;
        self
    }

    /// Generate all combinations of feature indices for polynomial terms
    fn generate_combinations(&self, n_features: usize) -> Vec<Vec<usize>> {
        let mut combinations = Vec::new();

        // Add bias term if requested
        if self.include_bias {
            combinations.push(vec![]);
        }

        // Generate combinations for each degree
        for degree in 1..=self.degree {
            self.generate_combinations_recursive(
                &mut combinations,
                &mut Vec::new(),
                0,
                n_features,
                degree,
            );
        }

        combinations
    }

    fn generate_combinations_recursive(
        &self,
        combinations: &mut Vec<Vec<usize>>,
        current: &mut Vec<usize>,
        start_idx: usize,
        n_features: usize,
        remaining_degree: usize,
    ) {
        if remaining_degree == 0 {
            combinations.push(current.clone());
            return;
        }

        for i in start_idx..n_features {
            current.push(i);
            let next_start = if self.interaction_only { i + 1 } else { i };
            self.generate_combinations_recursive(
                combinations,
                current,
                next_start,
                n_features,
                remaining_degree - 1,
            );
            current.pop();
        }
    }

    /// Compute polynomial feature value from a combination
    fn compute_polynomial_feature(&self, features: &[T], combination: &[usize]) -> T {
        if combination.is_empty() {
            return T::one(); // Bias term
        }

        let mut result = T::one();
        for &feature_idx in combination {
            if feature_idx < features.len() {
                result = result * features[feature_idx];
            }
        }
        result
    }
}

impl<T> Transform<T> for PolynomialFeatures<T>
where
    T: Clone
        + Default
        + num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn apply(&self, sample: (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)> {
        let (features, labels) = sample;
        let original_shape = features.shape().dims();
        let n_features = features.shape().size();

        // Flatten features for processing
        let feature_data = features.as_slice().ok_or_else(|| {
            TensorError::invalid_argument(
                "Cannot access tensor data (GPU tensor not supported)".to_string(),
            )
        })?;

        // Generate polynomial combinations
        let combinations = self.generate_combinations(n_features);
        let mut polynomial_data = Vec::with_capacity(combinations.len());

        // Compute polynomial features
        for combination in &combinations {
            let poly_value = self.compute_polynomial_feature(feature_data, combination);
            polynomial_data.push(poly_value);
        }

        let polynomial_features = Tensor::from_vec(polynomial_data, &[combinations.len()])?;
        Ok((polynomial_features, labels))
    }
}

/// Binning transformation for continuous features
/// Transforms continuous values into discrete bins
pub struct BinningTransform<T> {
    n_bins: usize,
    strategy: BinningStrategy,
    bin_edges: Option<Vec<T>>,
    _phantom: PhantomData<T>,
}

#[derive(Debug, Clone)]
pub enum BinningStrategy {
    /// Equal-width bins
    Uniform,
    /// Equal-frequency bins (quantiles)
    Quantile,
    /// K-means clustering for bin centers
    KMeans,
}

impl<T> BinningTransform<T>
where
    T: Clone
        + Default
        + num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    pub fn new(n_bins: usize, strategy: BinningStrategy) -> Self {
        Self {
            n_bins: n_bins.max(2),
            strategy,
            bin_edges: None,
            _phantom: PhantomData,
        }
    }

    pub fn uniform(n_bins: usize) -> Self {
        Self::new(n_bins, BinningStrategy::Uniform)
    }

    pub fn quantile(n_bins: usize) -> Self {
        Self::new(n_bins, BinningStrategy::Quantile)
    }

    pub fn with_edges(mut self, edges: Vec<T>) -> Self {
        self.bin_edges = Some(edges);
        self
    }

    /// Fit the binning transform to data
    pub fn fit(&mut self, data: &[T]) -> Result<()> {
        if data.is_empty() {
            return Err(TensorError::invalid_argument(
                "Cannot fit binning transform on empty data".to_string(),
            ));
        }

        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min_val = sorted_data[0];
        let max_val = sorted_data[sorted_data.len() - 1];

        let edges = match self.strategy {
            BinningStrategy::Uniform => {
                let mut edges = Vec::with_capacity(self.n_bins + 1);
                let step = (max_val - min_val) / T::from(self.n_bins).unwrap();

                for i in 0..=self.n_bins {
                    edges.push(min_val + T::from(i).unwrap() * step);
                }
                edges
            }
            BinningStrategy::Quantile => {
                let mut edges = Vec::with_capacity(self.n_bins + 1);
                edges.push(min_val);

                for i in 1..self.n_bins {
                    let quantile = i as f64 / self.n_bins as f64;
                    let idx = (quantile * (sorted_data.len() - 1) as f64) as usize;
                    edges.push(sorted_data[idx]);
                }

                edges.push(max_val);
                edges
            }
            BinningStrategy::KMeans => {
                // Simplified k-means for bin centers
                let mut centers = Vec::with_capacity(self.n_bins);
                let step = (max_val - min_val) / T::from(self.n_bins - 1).unwrap();

                for i in 0..self.n_bins {
                    centers.push(min_val + T::from(i).unwrap() * step);
                }

                // Convert centers to edges (midpoints)
                let mut edges = vec![min_val];
                for i in 1..self.n_bins {
                    let midpoint = (centers[i - 1] + centers[i]) / T::from(2.0).unwrap();
                    edges.push(midpoint);
                }
                edges.push(max_val);
                edges
            }
        };

        self.bin_edges = Some(edges);
        Ok(())
    }

    /// Find which bin a value belongs to
    fn find_bin(&self, value: T) -> usize {
        if let Some(ref edges) = self.bin_edges {
            for (i, &edge) in edges.iter().enumerate().skip(1) {
                if value <= edge {
                    return i - 1;
                }
            }
            return edges.len() - 2; // Last bin
        }
        0 // Default to first bin if not fitted
    }
}

impl<T> Transform<T> for BinningTransform<T>
where
    T: Clone
        + Default
        + num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn apply(&self, sample: (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)> {
        let (features, labels) = sample;

        if self.bin_edges.is_none() {
            return Err(TensorError::invalid_argument(
                "BinningTransform must be fitted before use".to_string(),
            ));
        }

        let feature_data = features.as_slice().ok_or_else(|| {
            TensorError::invalid_argument(
                "Cannot access tensor data (GPU tensor not supported)".to_string(),
            )
        })?;

        let binned_data: Vec<T> = feature_data
            .iter()
            .map(|&val| T::from(self.find_bin(val)).unwrap())
            .collect();

        let binned_features = Tensor::from_vec(binned_data, features.shape().dims())?;
        Ok((binned_features, labels))
    }
}

/// One-hot encoding transformation
/// Converts categorical features to binary feature vectors
pub struct OneHotEncode<T> {
    categories: Option<HashMap<usize, Vec<T>>>, // category value -> one-hot index mapping
    drop_first: bool,
    _phantom: PhantomData<T>,
}

impl<T> OneHotEncode<T>
where
    T: Clone
        + Default
        + num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable
        + std::hash::Hash
        + Eq,
{
    pub fn new() -> Self {
        Self {
            categories: None,
            drop_first: false,
            _phantom: PhantomData,
        }
    }

    pub fn drop_first(mut self) -> Self {
        self.drop_first = true;
        self
    }

    /// Fit the encoder to discover categories
    pub fn fit(&mut self, data: &[T]) -> Result<()> {
        let mut unique_values: Vec<T> = data.iter().cloned().collect();
        unique_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        unique_values.dedup();

        let mut categories = HashMap::new();
        categories.insert(0, unique_values);

        self.categories = Some(categories);
        Ok(())
    }

    /// Get the number of output features after one-hot encoding
    pub fn output_size(&self) -> usize {
        if let Some(ref categories) = self.categories {
            let base_size = categories.get(&0).map(|c| c.len()).unwrap_or(0);
            if self.drop_first && base_size > 0 {
                base_size - 1
            } else {
                base_size
            }
        } else {
            0
        }
    }
}

impl<T> Default for OneHotEncode<T>
where
    T: Clone
        + Default
        + num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable
        + std::hash::Hash
        + Eq,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Transform<T> for OneHotEncode<T>
where
    T: Clone
        + Default
        + num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable
        + std::hash::Hash
        + Eq,
{
    fn apply(&self, sample: (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)> {
        let (features, labels) = sample;

        if self.categories.is_none() {
            return Err(TensorError::invalid_argument(
                "OneHotEncode must be fitted before use".to_string(),
            ));
        }

        let feature_data = features.as_slice().ok_or_else(|| {
            TensorError::invalid_argument(
                "Cannot access tensor data (GPU tensor not supported)".to_string(),
            )
        })?;

        let categories = self.categories.as_ref().unwrap().get(&0).unwrap();
        let output_size = self.output_size();
        let mut encoded_data = Vec::with_capacity(output_size * feature_data.len());

        for &value in feature_data {
            let mut one_hot = vec![T::zero(); output_size];

            if let Some(pos) = categories.iter().position(|&cat| cat == value) {
                let adjusted_pos = if self.drop_first && pos > 0 {
                    pos - 1
                } else if self.drop_first && pos == 0 {
                    // First category is dropped, so no position to set
                    one_hot.len() // Invalid position, will be caught below
                } else {
                    pos
                };

                if adjusted_pos < one_hot.len() {
                    one_hot[adjusted_pos] = T::one();
                }
            }

            encoded_data.extend(one_hot);
        }

        let new_shape = vec![feature_data.len(), output_size];
        let encoded_features = Tensor::from_vec(encoded_data, &new_shape)?;
        Ok((encoded_features, labels))
    }
}

/// Target encoding transformation
/// Encodes categorical features using target statistics
pub struct TargetEncode<T> {
    category_means: Option<HashMap<T, T>>,
    global_mean: Option<T>,
    smoothing: f64,
    _phantom: PhantomData<T>,
}

impl<T> TargetEncode<T>
where
    T: Clone
        + Default
        + num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable
        + std::hash::Hash
        + Eq,
{
    pub fn new(smoothing: f64) -> Self {
        Self {
            category_means: None,
            global_mean: None,
            smoothing: smoothing.max(0.0),
            _phantom: PhantomData,
        }
    }

    /// Fit the encoder using features and targets
    pub fn fit(&mut self, features: &[T], targets: &[T]) -> Result<()> {
        if features.len() != targets.len() {
            return Err(TensorError::invalid_argument(
                "Features and targets must have the same length".to_string(),
            ));
        }

        // Calculate global mean
        let global_mean =
            targets.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(targets.len()).unwrap();
        self.global_mean = Some(global_mean);

        // Calculate category means
        let mut category_sums: HashMap<T, (T, usize)> = HashMap::new();

        for (&feature, &target) in features.iter().zip(targets.iter()) {
            let entry = category_sums.entry(feature).or_insert((T::zero(), 0));
            entry.0 = entry.0 + target;
            entry.1 += 1;
        }

        let mut category_means = HashMap::new();
        for (category, (sum, count)) in category_sums {
            let category_mean = sum / T::from(count).unwrap();

            // Apply smoothing
            let smoothed_mean = if self.smoothing > 0.0 {
                let alpha = T::from(self.smoothing).unwrap();
                let n = T::from(count).unwrap();
                (category_mean * n + global_mean * alpha) / (n + alpha)
            } else {
                category_mean
            };

            category_means.insert(category, smoothed_mean);
        }

        self.category_means = Some(category_means);
        Ok(())
    }
}

impl<T> Transform<T> for TargetEncode<T>
where
    T: Clone
        + Default
        + num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable
        + std::hash::Hash
        + Eq,
{
    fn apply(&self, sample: (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)> {
        let (features, labels) = sample;

        if self.category_means.is_none() || self.global_mean.is_none() {
            return Err(TensorError::invalid_argument(
                "TargetEncode must be fitted before use".to_string(),
            ));
        }

        let feature_data = features.as_slice().ok_or_else(|| {
            TensorError::invalid_argument(
                "Cannot access tensor data (GPU tensor not supported)".to_string(),
            )
        })?;

        let category_means = self.category_means.as_ref().unwrap();
        let global_mean = self.global_mean.unwrap();

        let encoded_data: Vec<T> = feature_data
            .iter()
            .map(|&value| category_means.get(&value).copied().unwrap_or(global_mean))
            .collect();

        let encoded_features = Tensor::from_vec(encoded_data, features.shape().dims())?;
        Ok((encoded_features, labels))
    }
}

/// Feature selection based on variance threshold
pub struct VarianceThreshold<T> {
    threshold: T,
    selected_features: Option<Vec<usize>>,
    _phantom: PhantomData<T>,
}

impl<T> VarianceThreshold<T>
where
    T: Clone
        + Default
        + num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    pub fn new(threshold: T) -> Self {
        Self {
            threshold,
            selected_features: None,
            _phantom: PhantomData,
        }
    }

    /// Fit the selector to identify features with sufficient variance
    pub fn fit(&mut self, data: &[Vec<T>]) -> Result<()> {
        if data.is_empty() || data[0].is_empty() {
            return Err(TensorError::invalid_argument(
                "Cannot fit on empty data".to_string(),
            ));
        }

        let n_features = data[0].len();
        let n_samples = T::from(data.len()).unwrap();
        let mut selected = Vec::new();

        for feature_idx in 0..n_features {
            // Calculate mean
            let mut sum = T::zero();
            for sample in data {
                if feature_idx < sample.len() {
                    sum = sum + sample[feature_idx];
                }
            }
            let mean = sum / n_samples;

            // Calculate variance
            let mut variance_sum = T::zero();
            for sample in data {
                if feature_idx < sample.len() {
                    let diff = sample[feature_idx] - mean;
                    variance_sum = variance_sum + diff * diff;
                }
            }
            let variance = variance_sum / n_samples;

            if variance >= self.threshold {
                selected.push(feature_idx);
            }
        }

        self.selected_features = Some(selected);
        Ok(())
    }

    /// Get the indices of selected features
    pub fn get_selected_features(&self) -> &Option<Vec<usize>> {
        &self.selected_features
    }
}

impl<T> Transform<T> for VarianceThreshold<T>
where
    T: Clone
        + Default
        + num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn apply(&self, sample: (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)> {
        let (features, labels) = sample;

        if self.selected_features.is_none() {
            return Err(TensorError::invalid_argument(
                "VarianceThreshold must be fitted before use".to_string(),
            ));
        }

        let feature_data = features.as_slice().ok_or_else(|| {
            TensorError::invalid_argument(
                "Cannot access tensor data (GPU tensor not supported)".to_string(),
            )
        })?;

        let selected = self.selected_features.as_ref().unwrap();
        let mut filtered_data = Vec::with_capacity(selected.len());

        for &idx in selected {
            if idx < feature_data.len() {
                filtered_data.push(feature_data[idx]);
            }
        }

        let filtered_features = Tensor::from_vec(filtered_data, &[selected.len()])?;
        Ok((filtered_features, labels))
    }
}

/// Feature scaling using power transformation
pub struct PowerTransform<T> {
    method: PowerMethod,
    fitted_lambdas: Option<Vec<T>>,
    _phantom: PhantomData<T>,
}

#[derive(Debug, Clone)]
pub enum PowerMethod {
    /// Box-Cox transformation (requires positive values)
    BoxCox,
    /// Yeo-Johnson transformation (handles negative values)
    YeoJohnson,
}

impl<T> PowerTransform<T>
where
    T: Clone
        + Default
        + num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    pub fn new(method: PowerMethod) -> Self {
        Self {
            method,
            fitted_lambdas: None,
            _phantom: PhantomData,
        }
    }

    pub fn box_cox() -> Self {
        Self::new(PowerMethod::BoxCox)
    }

    pub fn yeo_johnson() -> Self {
        Self::new(PowerMethod::YeoJohnson)
    }

    /// Fit the transformer to find optimal lambda parameters
    pub fn fit(&mut self, data: &[T]) -> Result<()> {
        // For simplicity, we'll use a fixed lambda of 0.5 for Box-Cox
        // and 1.0 for Yeo-Johnson. In practice, you'd optimize these.
        let lambda = match self.method {
            PowerMethod::BoxCox => T::from(0.5).unwrap(),
            PowerMethod::YeoJohnson => T::one(),
        };

        self.fitted_lambdas = Some(vec![lambda]);
        Ok(())
    }

    /// Apply Box-Cox transformation
    fn box_cox_transform(&self, value: T, lambda: T) -> T {
        if value <= T::zero() {
            return T::zero(); // Handle non-positive values
        }

        if lambda == T::zero() {
            value.ln()
        } else {
            let one = T::one();
            (value.powf(lambda) - one) / lambda
        }
    }

    /// Apply Yeo-Johnson transformation
    fn yeo_johnson_transform(&self, value: T, lambda: T) -> T {
        let one = T::one();
        let two = T::from(2.0).unwrap();

        if value >= T::zero() {
            if lambda == T::zero() {
                (value + one).ln()
            } else {
                ((value + one).powf(lambda) - one) / lambda
            }
        } else {
            if lambda == two {
                -((-value + one).ln())
            } else {
                -((((-value) + one).powf(two - lambda) - one) / (two - lambda))
            }
        }
    }
}

impl<T> Transform<T> for PowerTransform<T>
where
    T: Clone
        + Default
        + num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn apply(&self, sample: (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)> {
        let (features, labels) = sample;

        if self.fitted_lambdas.is_none() {
            return Err(TensorError::invalid_argument(
                "PowerTransform must be fitted before use".to_string(),
            ));
        }

        let feature_data = features.as_slice().ok_or_else(|| {
            TensorError::invalid_argument(
                "Cannot access tensor data (GPU tensor not supported)".to_string(),
            )
        })?;

        let lambda = self.fitted_lambdas.as_ref().unwrap()[0];
        let transformed_data: Vec<T> = feature_data
            .iter()
            .map(|&value| match self.method {
                PowerMethod::BoxCox => self.box_cox_transform(value, lambda),
                PowerMethod::YeoJohnson => self.yeo_johnson_transform(value, lambda),
            })
            .collect();

        let transformed_features = Tensor::from_vec(transformed_data, features.shape().dims())?;
        Ok((transformed_features, labels))
    }
}
