use crate::loss::{advanced_knowledge_distillation_loss, knowledge_distillation_loss};
use crate::optimizers::Optimizer;
use crate::Model;
use scirs2_core::num_traits::{Float, FromPrimitive, One, Zero};
use tenflowers_core::{Result, Tensor, TensorError};

/// Configuration for knowledge distillation training
#[derive(Debug, Clone)]
pub struct DistillationConfig<T> {
    /// Temperature for softening probability distributions (typical: 3-5)
    pub temperature: T,
    /// Weight for distillation loss vs hard target loss (typical: 0.5-0.9)  
    pub alpha: T,
    /// Weight for feature matching loss (if using advanced distillation)
    pub beta: T,
    /// Whether to use feature matching between intermediate layers
    pub feature_matching: bool,
    /// Whether to freeze teacher model during training
    pub freeze_teacher: bool,
    /// Learning rate for student model
    pub student_lr: T,
    /// Maximum number of training epochs
    pub max_epochs: usize,
    /// Batch size for training
    pub batch_size: usize,
}

impl<T> Default for DistillationConfig<T>
where
    T: Float + FromPrimitive,
{
    fn default() -> Self {
        Self {
            temperature: T::from_f32(4.0).unwrap(),
            alpha: T::from_f32(0.7).unwrap(),
            beta: T::from_f32(0.3).unwrap(),
            feature_matching: false,
            freeze_teacher: true,
            student_lr: T::from_f32(0.001).unwrap(),
            max_epochs: 100,
            batch_size: 32,
        }
    }
}

/// Metrics tracked during distillation training
#[derive(Debug, Clone)]
pub struct DistillationMetrics<T> {
    /// Total combined loss (distillation + hard target)
    pub total_loss: T,
    /// Distillation loss component
    pub distillation_loss: T,
    /// Hard target loss component
    pub hard_loss: T,
    /// Feature matching loss (if enabled)
    pub feature_loss: Option<T>,
    /// Student model accuracy on validation set
    pub student_accuracy: T,
    /// Teacher model accuracy on validation set (for comparison)
    pub teacher_accuracy: T,
    /// Current epoch number
    pub epoch: usize,
}

/// Knowledge Distillation Trainer
/// Implements teacher-student training with soft target guidance
pub struct DistillationTrainer<T, O> {
    /// Pre-trained teacher model (usually larger/more complex)
    teacher: Box<dyn Model<T>>,
    /// Student model to be trained (usually smaller/simpler)
    student: Box<dyn Model<T>>,
    /// Optimizer for student model parameters
    student_optimizer: O,
    /// Distillation training configuration
    config: DistillationConfig<T>,
    /// Training metrics history
    metrics_history: Vec<DistillationMetrics<T>>,
}

impl<T, O> DistillationTrainer<T, O>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Neg<Output = T>
        + std::ops::Div<Output = T>
        + std::iter::Sum
        + bytemuck::Pod
        + bytemuck::Zeroable,
    O: Optimizer<T>,
{
    /// Create a new distillation trainer
    pub fn new(
        mut teacher: Box<dyn Model<T>>,
        mut student: Box<dyn Model<T>>,
        student_optimizer: O,
        config: DistillationConfig<T>,
    ) -> Self {
        // Set teacher to evaluation mode if freezing is enabled
        if config.freeze_teacher {
            teacher.set_training(false);
        }

        // Set student to training mode
        student.set_training(true);

        Self {
            teacher,
            student,
            student_optimizer,
            config,
            metrics_history: Vec::new(),
        }
    }

    /// Train the student model using knowledge distillation
    pub fn train(
        &mut self,
        train_inputs: &[&Tensor<T>],
        train_targets: &[&Tensor<i64>],
        val_inputs: Option<&[&Tensor<T>]>,
        val_targets: Option<&[&Tensor<i64>]>,
    ) -> Result<()> {
        if train_inputs.len() != train_targets.len() {
            return Err(TensorError::invalid_argument(
                "Number of inputs and targets must match".to_string(),
            ));
        }

        let num_samples = train_inputs.len();
        let num_batches = (num_samples + self.config.batch_size - 1) / self.config.batch_size;

        for epoch in 0..self.config.max_epochs {
            let mut epoch_total_loss = T::zero();
            let mut epoch_distill_loss = T::zero();
            let mut epoch_hard_loss = T::zero();

            // Training loop
            for batch_idx in 0..num_batches {
                let batch_start = batch_idx * self.config.batch_size;
                let batch_end = std::cmp::min(batch_start + self.config.batch_size, num_samples);

                // Get batch data
                let batch_inputs = &train_inputs[batch_start..batch_end];
                let batch_targets = &train_targets[batch_start..batch_end];

                // Process each sample in the batch
                for (input, target) in batch_inputs.iter().zip(batch_targets.iter()) {
                    // Forward pass through teacher (no gradients needed)
                    let teacher_logits = if self.config.freeze_teacher {
                        // Teacher in eval mode
                        self.teacher.forward(input)?
                    } else {
                        self.teacher.forward(input)?
                    };

                    // Forward pass through student
                    let student_logits = self.student.forward(input)?;

                    // Compute distillation loss
                    let loss = if self.config.feature_matching {
                        // Advanced distillation with feature matching
                        // Extract intermediate features from both models
                        let teacher_features = self.teacher.extract_features(input)?;
                        let student_features = self.student.extract_features(input)?;

                        // Convert to the required format for the loss function
                        let (teacher_feature_refs, student_feature_refs) =
                            if let (Some(t_feats), Some(s_feats)) =
                                (&teacher_features, &student_features)
                            {
                                let t_refs: Vec<&Tensor<T>> = t_feats.iter().collect();
                                let s_refs: Vec<&Tensor<T>> = s_feats.iter().collect();
                                (Some(t_refs), Some(s_refs))
                            } else {
                                (None, None)
                            };

                        advanced_knowledge_distillation_loss(
                            &student_logits,
                            &teacher_logits,
                            student_feature_refs.as_deref(),
                            teacher_feature_refs.as_deref(),
                            target,
                            self.config.temperature,
                            self.config.alpha,
                            self.config.beta,
                        )?
                    } else {
                        // Standard knowledge distillation
                        knowledge_distillation_loss(
                            &student_logits,
                            &teacher_logits,
                            target,
                            self.config.temperature,
                            self.config.alpha,
                        )?
                    };

                    // Backward pass and optimization (for student only)
                    self.student.zero_grad();

                    // Compute gradients through backpropagation
                    loss.backward()?;

                    // Update student model parameters using a manual approach
                    // Since we can't pass Box<dyn Model<T>> to step(), we'll update parameters manually
                    {
                        let params = self.student.parameters_mut();
                        for param in params {
                            if let Some(grad) = param.grad() {
                                // Apply simple SGD update: param = param - lr * grad
                                let lr = T::from(self.config.student_lr).unwrap();
                                let update = grad.mul(&Tensor::from_scalar(lr))?;
                                let new_param = param.sub(&update)?;
                                *param = new_param;
                            }
                        }
                    }

                    // Accumulate losses for metrics
                    if let Some(loss_scalar) = loss.get(&[]) {
                        epoch_total_loss = epoch_total_loss + loss_scalar;

                        // Compute separate loss components for monitoring
                        // Hard target loss (standard cross-entropy)
                        let hard_loss_component =
                            crate::loss::sparse_categorical_cross_entropy(&student_logits, target)?;
                        if let Some(hard_scalar) = hard_loss_component.get(&[]) {
                            epoch_hard_loss = epoch_hard_loss + hard_scalar;
                        }

                        // Estimate distillation loss component
                        // Since total = alpha * distill + (1-alpha) * hard + beta * feature (if feature matching)
                        if self.config.alpha > T::zero() {
                            let one_minus_alpha = T::one() - self.config.alpha;
                            let base_distill_estimate = loss_scalar
                                - one_minus_alpha
                                    * hard_loss_component.get(&[]).unwrap_or(T::zero());

                            // If using feature matching, the remaining loss includes both distillation and feature components
                            if self.config.feature_matching {
                                // For simplicity, attribute the remaining loss to distillation
                                // In a more sophisticated implementation, we'd separate feature matching loss
                                epoch_distill_loss = epoch_distill_loss + base_distill_estimate;
                            } else {
                                let distill_estimate = base_distill_estimate / self.config.alpha;
                                epoch_distill_loss = epoch_distill_loss + distill_estimate;
                            }
                        }
                    }
                }
            }

            // Compute validation metrics if validation data provided
            let (student_acc, teacher_acc) =
                if let (Some(val_inputs), Some(val_targets)) = (val_inputs, val_targets) {
                    let student_acc =
                        self.compute_accuracy(self.student.as_ref(), val_inputs, val_targets)?;
                    let teacher_acc =
                        self.compute_accuracy(self.teacher.as_ref(), val_inputs, val_targets)?;
                    (student_acc, teacher_acc)
                } else {
                    (T::zero(), T::zero())
                };

            // Record metrics
            let feature_loss_avg = if self.config.feature_matching {
                // When feature matching is enabled, estimate the feature loss component
                Some(epoch_distill_loss / T::from(num_samples).unwrap() * self.config.beta)
            } else {
                None
            };

            let epoch_metrics = DistillationMetrics {
                total_loss: epoch_total_loss / T::from(num_samples).unwrap(),
                distillation_loss: epoch_distill_loss / T::from(num_samples).unwrap(),
                hard_loss: epoch_hard_loss / T::from(num_samples).unwrap(),
                feature_loss: feature_loss_avg,
                student_accuracy: student_acc,
                teacher_accuracy: teacher_acc,
                epoch,
            };

            self.metrics_history.push(epoch_metrics.clone());

            // Print progress
            if epoch % 10 == 0 {
                println!(
                    "Epoch {}: Loss={:.4}, Student Acc={:.4}, Teacher Acc={:.4}",
                    epoch,
                    epoch_metrics.total_loss.to_f32().unwrap_or(0.0),
                    epoch_metrics.student_accuracy.to_f32().unwrap_or(0.0),
                    epoch_metrics.teacher_accuracy.to_f32().unwrap_or(0.0)
                );
            }
        }

        Ok(())
    }

    /// Compute accuracy on a validation set
    fn compute_accuracy(
        &self,
        model: &dyn Model<T>,
        inputs: &[&Tensor<T>],
        targets: &[&Tensor<i64>],
    ) -> Result<T> {
        if inputs.len() != targets.len() {
            return Err(TensorError::invalid_argument(
                "Number of inputs and targets must match".to_string(),
            ));
        }

        let mut correct = 0;
        let total = inputs.len();

        for (input, target) in inputs.iter().zip(targets.iter()) {
            let logits = model.forward(input)?;

            // Get predicted class (argmax of logits)
            let predicted = self.argmax(&logits)?;
            let true_label = target.get(&[0]).ok_or_else(|| {
                TensorError::invalid_argument("Could not get target label".to_string())
            })?;

            if predicted == true_label {
                correct += 1;
            }
        }

        Ok(T::from(correct).unwrap() / T::from(total).unwrap())
    }

    /// Helper function to compute argmax of a tensor
    fn argmax(&self, tensor: &Tensor<T>) -> Result<i64> {
        // Simple argmax implementation
        // In a full implementation, this would be in tenflowers_core::ops
        let shape = tensor.shape().dims();
        let num_classes = shape[shape.len() - 1];

        let mut max_val = T::neg_infinity();
        let mut max_idx = 0i64;

        for i in 0..num_classes {
            let mut idx = vec![0; shape.len()];
            idx[shape.len() - 1] = i;

            if let Some(val) = tensor.get(&idx) {
                if val > max_val {
                    max_val = val;
                    max_idx = i as i64;
                }
            }
        }

        Ok(max_idx)
    }

    /// Get the trained student model
    pub fn student_model(&self) -> &dyn Model<T> {
        self.student.as_ref()
    }

    /// Get training metrics history
    pub fn metrics_history(&self) -> &[DistillationMetrics<T>] {
        &self.metrics_history
    }

    /// Get the current distillation configuration
    pub fn config(&self) -> &DistillationConfig<T> {
        &self.config
    }

    /// Update distillation configuration
    pub fn set_config(&mut self, config: DistillationConfig<T>) {
        self.config = config;
    }
}

/// Builder for creating distillation trainers with common configurations
pub struct DistillationTrainerBuilder<T, O> {
    teacher: Option<Box<dyn Model<T>>>,
    student: Option<Box<dyn Model<T>>>,
    student_optimizer: Option<O>,
    config: DistillationConfig<T>,
}

impl<T, O> DistillationTrainerBuilder<T, O>
where
    T: Float
        + FromPrimitive
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Neg<Output = T>
        + std::ops::Div<Output = T>
        + std::iter::Sum
        + bytemuck::Pod
        + bytemuck::Zeroable,
    O: Optimizer<T>,
{
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            teacher: None,
            student: None,
            student_optimizer: None,
            config: DistillationConfig::default(),
        }
    }

    /// Set the teacher model
    pub fn with_teacher(mut self, teacher: Box<dyn Model<T>>) -> Self {
        self.teacher = Some(teacher);
        self
    }

    /// Set the student model
    pub fn with_student(mut self, student: Box<dyn Model<T>>) -> Self {
        self.student = Some(student);
        self
    }

    /// Set the optimizer for the student model
    pub fn with_optimizer(mut self, optimizer: O) -> Self {
        self.student_optimizer = Some(optimizer);
        self
    }

    /// Set the temperature for distillation
    pub fn with_temperature(mut self, temperature: T) -> Self {
        self.config.temperature = temperature;
        self
    }

    /// Set the alpha weight for distillation vs hard loss
    pub fn with_alpha(mut self, alpha: T) -> Self {
        self.config.alpha = alpha;
        self
    }

    /// Enable feature matching with given weight
    pub fn with_feature_matching(mut self, beta: T) -> Self {
        self.config.feature_matching = true;
        self.config.beta = beta;
        self
    }

    /// Set the learning rate for student training
    pub fn with_learning_rate(mut self, lr: T) -> Self {
        self.config.student_lr = lr;
        self
    }

    /// Set the maximum number of training epochs
    pub fn with_max_epochs(mut self, epochs: usize) -> Self {
        self.config.max_epochs = epochs;
        self
    }

    /// Set the batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    /// Build the distillation trainer
    pub fn build(self) -> Result<DistillationTrainer<T, O>> {
        let teacher = self.teacher.ok_or_else(|| {
            TensorError::invalid_argument("Teacher model is required".to_string())
        })?;

        let student = self.student.ok_or_else(|| {
            TensorError::invalid_argument("Student model is required".to_string())
        })?;

        let optimizer = self.student_optimizer.ok_or_else(|| {
            TensorError::invalid_argument("Student optimizer is required".to_string())
        })?;

        Ok(DistillationTrainer::new(
            teacher,
            student,
            optimizer,
            self.config,
        ))
    }
}

impl<T, O> Default for DistillationTrainerBuilder<T, O>
where
    T: Float
        + FromPrimitive
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Neg<Output = T>
        + std::ops::Div<Output = T>
        + std::iter::Sum
        + bytemuck::Pod
        + bytemuck::Zeroable,
    O: Optimizer<T>,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to create a basic distillation trainer
pub fn create_distillation_trainer<T, O>(
    teacher: Box<dyn Model<T>>,
    student: Box<dyn Model<T>>,
    optimizer: O,
) -> Result<DistillationTrainer<T, O>>
where
    T: Float
        + FromPrimitive
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Neg<Output = T>
        + std::ops::Div<Output = T>
        + std::iter::Sum
        + bytemuck::Pod
        + bytemuck::Zeroable,
    O: Optimizer<T>,
{
    DistillationTrainerBuilder::new()
        .with_teacher(teacher)
        .with_student(student)
        .with_optimizer(optimizer)
        .build()
}

/// Convenience function to create a distillation trainer with custom temperature
pub fn create_distillation_trainer_with_temperature<T, O>(
    teacher: Box<dyn Model<T>>,
    student: Box<dyn Model<T>>,
    optimizer: O,
    temperature: T,
    alpha: T,
) -> Result<DistillationTrainer<T, O>>
where
    T: Float
        + FromPrimitive
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Neg<Output = T>
        + std::ops::Div<Output = T>
        + std::iter::Sum
        + bytemuck::Pod
        + bytemuck::Zeroable,
    O: Optimizer<T>,
{
    DistillationTrainerBuilder::new()
        .with_teacher(teacher)
        .with_student(student)
        .with_optimizer(optimizer)
        .with_temperature(temperature)
        .with_alpha(alpha)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Dense;
    use crate::optimizers::SGD;
    use tenflowers_core::Tensor;

    // Note: These are skeleton tests. In a full implementation,
    // they would test actual distillation training

    #[test]
    fn test_distillation_config_default() {
        let config = DistillationConfig::<f32>::default();
        assert!(config.temperature > 0.0);
        assert!(config.alpha >= 0.0 && config.alpha <= 1.0);
        assert!(!config.feature_matching);
        assert!(config.freeze_teacher);
    }

    #[test]
    fn test_distillation_trainer_builder() {
        // This test would create actual models in a full implementation
        // For now, we just test the builder pattern
        let builder = DistillationTrainerBuilder::<f32, crate::optimizers::SGD<f32>>::new()
            .with_temperature(3.0)
            .with_alpha(0.8)
            .with_learning_rate(0.001);

        assert_eq!(builder.config.temperature, 3.0);
        assert_eq!(builder.config.alpha, 0.8);
        assert_eq!(builder.config.student_lr, 0.001);
    }

    #[test]
    fn test_distillation_metrics() {
        let metrics = DistillationMetrics {
            total_loss: 0.5,
            distillation_loss: 0.3,
            hard_loss: 0.2,
            feature_loss: None,
            student_accuracy: 0.85,
            teacher_accuracy: 0.92,
            epoch: 10,
        };

        assert_eq!(metrics.epoch, 10);
        assert_eq!(metrics.student_accuracy, 0.85);
        assert!(metrics.feature_loss.is_none());
    }
}
