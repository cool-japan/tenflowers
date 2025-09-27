//! Transform pipeline and composition utilities
//!
//! This module provides utilities for composing multiple transforms into
//! pipelines, including lazy evaluation, parallel execution, and conditional transforms.

use crate::transforms::Transform;
use scirs2_core::random::Rng;
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::Arc;
use tenflowers_core::{Result, Tensor, TensorError};

/// Implement Transform for Box<dyn Transform<T> + Send + Sync> to enable trait object usage
impl<T> Transform<T> for Box<dyn Transform<T> + Send + Sync> {
    fn apply(&self, sample: (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)> {
        self.as_ref().apply(sample)
    }
}

/// Implement Transform for Box<dyn Transform<T>> to enable trait object usage
impl<T> Transform<T> for Box<dyn Transform<T>> {
    fn apply(&self, sample: (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)> {
        self.as_ref().apply(sample)
    }
}

/// A pipeline stage that can be executed conditionally
#[derive(Clone)]
pub enum PipelineStage<T, Tr: Transform<T>> {
    /// Always execute this transform
    Always(Tr),
    /// Execute with a given probability
    Conditional { transform: Tr, probability: f32 },
    /// Execute only if a condition is met
    Predicate {
        transform: Tr,
        condition: Arc<dyn Fn(&(Tensor<T>, Tensor<T>)) -> bool + Send + Sync>,
    },
}

impl<T, Tr: Transform<T>> PipelineStage<T, Tr> {
    pub fn always(transform: Tr) -> Self {
        Self::Always(transform)
    }

    pub fn conditional(transform: Tr, probability: f32) -> Self {
        Self::Conditional {
            transform,
            probability: probability.clamp(0.0, 1.0),
        }
    }

    pub fn predicate(
        transform: Tr,
        condition: impl Fn(&(Tensor<T>, Tensor<T>)) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self::Predicate {
            transform,
            condition: Arc::new(condition),
        }
    }

    /// Execute the stage if conditions are met
    pub fn execute(&self, sample: (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)>
    where
        T: Clone,
    {
        match self {
            Self::Always(transform) => transform.apply(sample),
            Self::Conditional {
                transform,
                probability,
            } => {
                let mut rng = scirs2_core::random::rng();
                if rng.random::<f32>() < *probability {
                    transform.apply(sample)
                } else {
                    Ok(sample)
                }
            }
            Self::Predicate {
                transform,
                condition,
            } => {
                if condition(&sample) {
                    transform.apply(sample)
                } else {
                    Ok(sample)
                }
            }
        }
    }
}

/// Sequential composition of transforms
pub struct Compose<T> {
    stages: Vec<Box<dyn Transform<T> + Send + Sync>>,
    _phantom: PhantomData<T>,
}

impl<T> Compose<T>
where
    T: Clone + Send + Sync + 'static,
{
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            _phantom: PhantomData,
        }
    }

    /// Add a transform to the pipeline
    pub fn add<Tr: Transform<T> + Send + Sync + 'static>(mut self, transform: Tr) -> Self {
        self.stages.push(Box::new(transform));
        self
    }

    /// Add a transform with a probability of execution
    pub fn add_conditional<Tr: Transform<T> + Send + Sync + 'static>(
        self,
        transform: Tr,
        probability: f32,
    ) -> Self {
        let stage = ConditionalTransform::new(transform, probability);
        self.add(stage)
    }

    /// Get the number of stages in the pipeline
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    /// Check if the pipeline is empty
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// Create a compose from a vector of transforms
    pub fn from_transforms(transforms: Vec<Box<dyn Transform<T> + Send + Sync>>) -> Self {
        Self {
            stages: transforms,
            _phantom: PhantomData,
        }
    }
}

impl<T> Default for Compose<T>
where
    T: Clone + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Transform<T> for Compose<T>
where
    T: Clone + Send + Sync + 'static,
{
    fn apply(&self, mut sample: (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)> {
        for stage in &self.stages {
            sample = stage.apply(sample)?;
        }
        Ok(sample)
    }
}

/// Lazy composition that only evaluates transforms when needed
pub struct LazyCompose<T> {
    stages: VecDeque<Box<dyn Transform<T> + Send + Sync>>,
    cache_results: bool,
    cached_sample: Option<(Tensor<T>, Tensor<T>)>,
    _phantom: PhantomData<T>,
}

impl<T> LazyCompose<T> {
    pub fn new() -> Self {
        Self {
            stages: VecDeque::new(),
            cache_results: false,
            cached_sample: None,
            _phantom: PhantomData,
        }
    }

    pub fn with_caching(mut self) -> Self {
        self.cache_results = true;
        self
    }

    /// Add a transform to the front of the pipeline
    pub fn prepend<Tr: Transform<T> + Send + Sync + 'static>(mut self, transform: Tr) -> Self {
        self.stages.push_front(Box::new(transform));
        self.invalidate_cache();
        self
    }

    /// Add a transform to the end of the pipeline
    pub fn append<Tr: Transform<T> + Send + Sync + 'static>(mut self, transform: Tr) -> Self {
        self.stages.push_back(Box::new(transform));
        self.invalidate_cache();
        self
    }

    /// Remove the last transform
    pub fn pop(&mut self) -> Option<Box<dyn Transform<T> + Send + Sync>> {
        self.invalidate_cache();
        self.stages.pop_back()
    }

    /// Remove the first transform
    pub fn pop_front(&mut self) -> Option<Box<dyn Transform<T> + Send + Sync>> {
        self.invalidate_cache();
        self.stages.pop_front()
    }

    fn invalidate_cache(&mut self) {
        if self.cache_results {
            self.cached_sample = None;
        }
    }

    /// Get the number of stages
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    /// Check if pipeline is empty
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }
}

impl<T> Default for LazyCompose<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Transform<T> for LazyCompose<T>
where
    T: Clone,
{
    fn apply(&self, sample: (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)> {
        // Check cache first
        if self.cache_results {
            if let Some(ref cached) = self.cached_sample {
                return Ok(cached.clone());
            }
        }

        let mut current_sample = sample;
        for stage in &self.stages {
            current_sample = stage.apply(current_sample)?;
        }

        // Cache result if caching is enabled
        if self.cache_results {
            // Note: This would require interior mutability in a real implementation
            // For now, we'll skip caching in apply() method
        }

        Ok(current_sample)
    }
}

/// Transform pipeline with advanced features
pub struct TransformPipeline<T> {
    stages: Vec<PipelineStage<T, Box<dyn Transform<T> + Send + Sync>>>,
    parallel_execution: bool,
    error_handling: ErrorHandlingStrategy,
    _phantom: PhantomData<T>,
}

#[derive(Debug, Clone)]
pub enum ErrorHandlingStrategy {
    /// Stop pipeline on first error
    Fail,
    /// Skip transforms that error and continue
    Skip,
    /// Use a fallback value on error
    Fallback,
}

impl<T> TransformPipeline<T> {
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            parallel_execution: false,
            error_handling: ErrorHandlingStrategy::Fail,
            _phantom: PhantomData,
        }
    }

    pub fn with_parallel_execution(mut self) -> Self {
        self.parallel_execution = true;
        self
    }

    pub fn with_error_handling(mut self, strategy: ErrorHandlingStrategy) -> Self {
        self.error_handling = strategy;
        self
    }

    /// Add a transform that always executes
    pub fn add<Tr: Transform<T> + Send + Sync + 'static>(mut self, transform: Tr) -> Self {
        let stage =
            PipelineStage::Always(Box::new(transform) as Box<dyn Transform<T> + Send + Sync>);
        self.stages.push(stage);
        self
    }

    /// Add a conditional transform
    pub fn add_conditional<Tr: Transform<T> + Send + Sync + 'static>(
        mut self,
        transform: Tr,
        probability: f32,
    ) -> Self {
        let stage = PipelineStage::Conditional {
            transform: Box::new(transform) as Box<dyn Transform<T> + Send + Sync>,
            probability,
        };
        self.stages.push(stage);
        self
    }

    /// Add a transform with a predicate
    pub fn add_predicate<Tr: Transform<T> + Send + Sync + 'static>(
        mut self,
        transform: Tr,
        condition: impl Fn(&(Tensor<T>, Tensor<T>)) -> bool + Send + Sync + 'static,
    ) -> Self {
        let stage = PipelineStage::Predicate {
            transform: Box::new(transform) as Box<dyn Transform<T> + Send + Sync>,
            condition: Arc::new(condition),
        };
        self.stages.push(stage);
        self
    }

    /// Get the number of stages
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    /// Check if pipeline is empty
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// Execute with error handling
    fn execute_with_error_handling(
        &self,
        mut sample: (Tensor<T>, Tensor<T>),
    ) -> Result<(Tensor<T>, Tensor<T>)>
    where
        T: Clone,
    {
        for stage in &self.stages {
            match stage.execute(sample.clone()) {
                Ok(result) => sample = result,
                Err(e) => match self.error_handling {
                    ErrorHandlingStrategy::Fail => return Err(e),
                    ErrorHandlingStrategy::Skip => {
                        // Continue with original sample
                        continue;
                    }
                    ErrorHandlingStrategy::Fallback => {
                        // Use original sample as fallback
                        continue;
                    }
                },
            }
        }
        Ok(sample)
    }
}

impl<T> Default for TransformPipeline<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Transform<T> for TransformPipeline<T>
where
    T: Clone,
{
    fn apply(&self, sample: (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)> {
        if self.parallel_execution {
            // For parallel execution, we'd need to use threads or async
            // For now, we'll fall back to sequential execution
            self.execute_with_error_handling(sample)
        } else {
            self.execute_with_error_handling(sample)
        }
    }
}

/// Conditional transform wrapper
pub struct ConditionalTransform<T, Tr: Transform<T>> {
    transform: Tr,
    probability: f32,
    _phantom: PhantomData<T>,
}

impl<T, Tr: Transform<T>> ConditionalTransform<T, Tr> {
    pub fn new(transform: Tr, probability: f32) -> Self {
        Self {
            transform,
            probability: probability.clamp(0.0, 1.0),
            _phantom: PhantomData,
        }
    }
}

impl<T, Tr: Transform<T>> Transform<T> for ConditionalTransform<T, Tr>
where
    T: Clone,
{
    fn apply(&self, sample: (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)> {
        let mut rng = scirs2_core::random::rng();
        if rng.random::<f32>() < self.probability {
            self.transform.apply(sample)
        } else {
            Ok(sample)
        }
    }
}

/// Random choice between multiple transforms
pub struct RandomChoice<T> {
    transforms: Vec<Box<dyn Transform<T> + Send + Sync>>,
    weights: Vec<f32>,
    _phantom: PhantomData<T>,
}

impl<T> RandomChoice<T> {
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
            weights: Vec::new(),
            _phantom: PhantomData,
        }
    }

    /// Add a transform with equal weight
    pub fn add<Tr: Transform<T> + Send + Sync + 'static>(mut self, transform: Tr) -> Self {
        self.transforms.push(Box::new(transform));
        self.weights.push(1.0);
        self
    }

    /// Add a transform with specific weight
    pub fn add_weighted<Tr: Transform<T> + Send + Sync + 'static>(
        mut self,
        transform: Tr,
        weight: f32,
    ) -> Self {
        self.transforms.push(Box::new(transform));
        self.weights.push(weight.max(0.0));
        self
    }

    /// Select a random transform based on weights
    fn select_transform(&self) -> Result<&Box<dyn Transform<T> + Send + Sync>> {
        if self.transforms.is_empty() {
            return Err(TensorError::invalid_argument(
                "RandomChoice has no transforms".to_string(),
            ));
        }

        let total_weight: f32 = self.weights.iter().sum();
        if total_weight <= 0.0 {
            return Err(TensorError::invalid_argument(
                "Total weight must be positive".to_string(),
            ));
        }

        let mut rng = scirs2_core::random::rng();
        let mut random_weight = rng.random::<f32>() * total_weight;

        for (i, &weight) in self.weights.iter().enumerate() {
            random_weight -= weight;
            if random_weight <= 0.0 {
                return Ok(&self.transforms[i]);
            }
        }

        // Fallback to last transform
        Ok(self.transforms.last().unwrap())
    }

    /// Get the number of transforms
    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }
}

impl<T> Default for RandomChoice<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Transform<T> for RandomChoice<T> {
    fn apply(&self, sample: (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)> {
        let selected_transform = self.select_transform()?;
        selected_transform.apply(sample)
    }
}

/// Branch transform - applies different transforms based on conditions
pub struct Branch<T> {
    branches: Vec<(
        Arc<dyn Fn(&(Tensor<T>, Tensor<T>)) -> bool + Send + Sync>,
        Box<dyn Transform<T> + Send + Sync>,
    )>,
    default_transform: Option<Box<dyn Transform<T> + Send + Sync>>,
    _phantom: PhantomData<T>,
}

impl<T> Branch<T> {
    pub fn new() -> Self {
        Self {
            branches: Vec::new(),
            default_transform: None,
            _phantom: PhantomData,
        }
    }

    /// Add a conditional branch
    pub fn add_branch<Tr: Transform<T> + Send + Sync + 'static>(
        mut self,
        condition: impl Fn(&(Tensor<T>, Tensor<T>)) -> bool + Send + Sync + 'static,
        transform: Tr,
    ) -> Self {
        self.branches
            .push((Arc::new(condition), Box::new(transform)));
        self
    }

    /// Set default transform when no conditions match
    pub fn default<Tr: Transform<T> + Send + Sync + 'static>(mut self, transform: Tr) -> Self {
        self.default_transform = Some(Box::new(transform));
        self
    }

    /// Find the first matching branch
    fn find_matching_branch(
        &self,
        sample: &(Tensor<T>, Tensor<T>),
    ) -> Option<&Box<dyn Transform<T> + Send + Sync>> {
        for (condition, transform) in &self.branches {
            if condition(sample) {
                return Some(transform);
            }
        }
        self.default_transform.as_ref()
    }
}

impl<T> Default for Branch<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Transform<T> for Branch<T>
where
    T: Clone,
{
    fn apply(&self, sample: (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)> {
        if let Some(transform) = self.find_matching_branch(&sample) {
            transform.apply(sample)
        } else {
            Ok(sample) // No matching branch, return original
        }
    }
}

/// Macro for easy pipeline composition
#[macro_export]
macro_rules! compose {
    ($($transform:expr),* $(,)?) => {
        {
            let mut pipeline = Compose::new();
            $(
                pipeline = pipeline.add($transform);
            )*
            pipeline
        }
    };
}

/// Macro for conditional transforms
#[macro_export]
macro_rules! conditional {
    ($transform:expr, $prob:expr) => {
        ConditionalTransform::new($transform, $prob)
    };
}

/// Macro for random choice
#[macro_export]
macro_rules! random_choice {
    ($($transform:expr),* $(,)?) => {
        {
            let mut choice = RandomChoice::new();
            $(
                choice = choice.add($transform);
            )*
            choice
        }
    };
    ($($transform:expr => $weight:expr),* $(,)?) => {
        {
            let mut choice = RandomChoice::new();
            $(
                choice = choice.add_weighted($transform, $weight);
            )*
            choice
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transforms::noise::AddNoise;

    #[test]
    fn test_compose() {
        let pipeline = Compose::new()
            .add(AddNoise::new(0.1f32))
            .add(AddNoise::new(0.2f32));

        let features = Tensor::<f32>::zeros(&[10]);
        let labels = Tensor::<f32>::zeros(&[1]);

        let result = pipeline.apply((features, labels));
        assert!(result.is_ok());
    }

    #[test]
    fn test_conditional_transform() {
        let transform = ConditionalTransform::new(AddNoise::new(0.1f32), 1.0);

        let features = Tensor::<f32>::zeros(&[10]);
        let labels = Tensor::<f32>::zeros(&[1]);

        let result = transform.apply((features, labels));
        assert!(result.is_ok());
    }

    #[test]
    fn test_random_choice() {
        let choice = RandomChoice::new()
            .add(AddNoise::new(0.1f32))
            .add(AddNoise::new(0.2f32));

        let features = Tensor::<f32>::zeros(&[10]);
        let labels = Tensor::<f32>::zeros(&[1]);

        let result = choice.apply((features, labels));
        assert!(result.is_ok());
    }

    #[test]
    fn test_pipeline_stage() {
        let stage = PipelineStage::always(AddNoise::new(0.1f32));

        let features = Tensor::<f32>::zeros(&[10]);
        let labels = Tensor::<f32>::zeros(&[1]);

        let result = stage.execute((features, labels));
        assert!(result.is_ok());
    }

    #[test]
    fn test_branch() {
        let branch = Branch::new()
            .add_branch(|sample| sample.0.shape().size() > 5, AddNoise::new(0.1f32))
            .default(AddNoise::new(0.2f32));

        let features = Tensor::<f32>::zeros(&[10]);
        let labels = Tensor::<f32>::zeros(&[1]);

        let result = branch.apply((features, labels));
        assert!(result.is_ok());
    }
}
