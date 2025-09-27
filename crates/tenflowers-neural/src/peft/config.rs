//! Configuration and utilities for Parameter-Efficient Fine-Tuning methods

use super::{
    adalora::AdaLoRAConfig, ia3::IA3Config, lora::LoRAConfig, prefix_tuning::PrefixTuningConfig,
    ptuning_v2::PTuningV2Config, qlora::QLoRAConfig,
};

/// Enumeration of supported PEFT methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PEFTMethod {
    /// LoRA (Low-Rank Adaptation)
    LoRA,
    /// QLoRA (Quantized LoRA) - Future implementation
    QLoRA,
    /// AdaLoRA (Adaptive LoRA) - Future implementation  
    AdaLoRA,
    /// Prefix Tuning - Future implementation
    PrefixTuning,
    /// P-Tuning v2 - Future implementation
    PTuningV2,
    /// IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations) - Future implementation
    IA3,
    /// Custom adapter implementation
    Custom(String),
}

impl PEFTMethod {
    /// Get a human-readable name for the PEFT method
    pub fn name(&self) -> &str {
        match self {
            PEFTMethod::LoRA => "LoRA",
            PEFTMethod::QLoRA => "QLoRA",
            PEFTMethod::AdaLoRA => "AdaLoRA",
            PEFTMethod::PrefixTuning => "Prefix Tuning",
            PEFTMethod::PTuningV2 => "P-Tuning v2",
            PEFTMethod::IA3 => "IA³",
            PEFTMethod::Custom(name) => name,
        }
    }

    /// Get a description of the PEFT method
    pub fn description(&self) -> &str {
        match self {
            PEFTMethod::LoRA => "Low-rank adaptation with trainable decomposition matrices",
            PEFTMethod::QLoRA => "Quantized LoRA for memory-efficient fine-tuning",
            PEFTMethod::AdaLoRA => "Adaptive LoRA with dynamic rank allocation",
            PEFTMethod::PrefixTuning => "Learnable prefix embeddings for language models",
            PEFTMethod::PTuningV2 => "Deep prompt tuning with learnable embeddings",
            PEFTMethod::IA3 => "Activation scaling without additional parameters",
            PEFTMethod::Custom(name) => "Custom adapter implementation",
        }
    }

    /// Check if the method is currently implemented
    pub fn is_implemented(&self) -> bool {
        matches!(
            self,
            PEFTMethod::LoRA
                | PEFTMethod::QLoRA
                | PEFTMethod::AdaLoRA
                | PEFTMethod::IA3
                | PEFTMethod::PrefixTuning
                | PEFTMethod::PTuningV2
                | PEFTMethod::Custom(_)
        )
    }
}

/// General configuration for PEFT methods
#[derive(Debug, Clone)]
pub struct PEFTConfig {
    /// The PEFT method to use
    pub method: PEFTMethod,
    /// Method-specific configuration
    pub method_config: MethodConfig,
    /// Whether to freeze the base model parameters
    pub freeze_base: bool,
    /// Target modules to apply PEFT (e.g., ["attention.query", "attention.value"])
    pub target_modules: Vec<String>,
    /// Modules to exclude from PEFT adaptation
    pub exclude_modules: Vec<String>,
    /// Whether to save only the adapter weights
    pub save_only_adapter: bool,
}

/// Method-specific configuration enum
#[derive(Debug, Clone)]
pub enum MethodConfig {
    /// LoRA-specific configuration
    LoRA(LoRAConfig),
    /// QLoRA-specific configuration
    QLoRA(QLoRAConfig),
    /// AdaLoRA-specific configuration
    AdaLoRA(AdaLoRAConfig),
    /// IA³-specific configuration
    IA3(IA3Config),
    /// Prefix Tuning-specific configuration
    PrefixTuning(PrefixTuningConfig),
    /// P-Tuning v2-specific configuration
    PTuningV2(PTuningV2Config),
    /// Placeholder for future methods
    Other(String),
}

impl PEFTConfig {
    /// Create a new PEFT configuration with LoRA
    pub fn lora(lora_config: LoRAConfig) -> Self {
        Self {
            method: PEFTMethod::LoRA,
            method_config: MethodConfig::LoRA(lora_config),
            freeze_base: true,
            target_modules: vec!["dense".to_string(), "linear".to_string()],
            exclude_modules: vec![],
            save_only_adapter: true,
        }
    }

    /// Create a new PEFT configuration with QLoRA
    pub fn qlora(qlora_config: QLoRAConfig) -> Self {
        Self {
            method: PEFTMethod::QLoRA,
            method_config: MethodConfig::QLoRA(qlora_config),
            freeze_base: true,
            target_modules: vec!["dense".to_string(), "linear".to_string()],
            exclude_modules: vec![],
            save_only_adapter: true,
        }
    }

    /// Create a new PEFT configuration with AdaLoRA
    pub fn adalora(adalora_config: AdaLoRAConfig) -> Self {
        Self {
            method: PEFTMethod::AdaLoRA,
            method_config: MethodConfig::AdaLoRA(adalora_config),
            freeze_base: true,
            target_modules: vec!["dense".to_string(), "linear".to_string()],
            exclude_modules: vec![],
            save_only_adapter: true,
        }
    }

    /// Create a new PEFT configuration with IA³
    pub fn ia3(ia3_config: IA3Config) -> Self {
        Self {
            method: PEFTMethod::IA3,
            method_config: MethodConfig::IA3(ia3_config),
            freeze_base: true,
            target_modules: vec!["attention".to_string(), "feedforward".to_string()],
            exclude_modules: vec![],
            save_only_adapter: true,
        }
    }

    /// Create a new PEFT configuration with Prefix Tuning
    pub fn prefix_tuning(prefix_config: PrefixTuningConfig) -> Self {
        Self {
            method: PEFTMethod::PrefixTuning,
            method_config: MethodConfig::PrefixTuning(prefix_config),
            freeze_base: true,
            target_modules: vec!["attention".to_string()], // Prefix tuning mainly affects attention
            exclude_modules: vec![],
            save_only_adapter: true,
        }
    }

    /// Create a new PEFT configuration with P-Tuning v2
    pub fn ptuning_v2(ptuning_config: PTuningV2Config) -> Self {
        Self {
            method: PEFTMethod::PTuningV2,
            method_config: MethodConfig::PTuningV2(ptuning_config),
            freeze_base: true,
            target_modules: vec!["transformer".to_string(), "layer".to_string()], // P-Tuning v2 affects multiple layers
            exclude_modules: vec![],
            save_only_adapter: true,
        }
    }

    /// Set target modules for PEFT adaptation
    pub fn with_target_modules(mut self, modules: Vec<String>) -> Self {
        self.target_modules = modules;
        self
    }

    /// Set modules to exclude from adaptation
    pub fn with_exclude_modules(mut self, modules: Vec<String>) -> Self {
        self.exclude_modules = modules;
        self
    }

    /// Set whether to freeze base model parameters
    pub fn with_freeze_base(mut self, freeze: bool) -> Self {
        self.freeze_base = freeze;
        self
    }

    /// Set whether to save only adapter weights
    pub fn with_save_only_adapter(mut self, save_only: bool) -> Self {
        self.save_only_adapter = save_only;
        self
    }

    /// Check if a module name should be adapted based on target/exclude lists
    pub fn should_adapt_module(&self, module_name: &str) -> bool {
        // Check exclude list first
        if self
            .exclude_modules
            .iter()
            .any(|exclude| module_name.contains(exclude))
        {
            return false;
        }

        // Check if it matches any target modules
        if self.target_modules.is_empty() {
            // If no target modules specified, adapt all (except excluded)
            true
        } else {
            self.target_modules
                .iter()
                .any(|target| module_name.contains(target))
        }
    }

    /// Get the LoRA configuration if this is a LoRA config
    pub fn lora_config(&self) -> Option<&LoRAConfig> {
        match &self.method_config {
            MethodConfig::LoRA(config) => Some(config),
            MethodConfig::QLoRA(config) => Some(&config.lora_config),
            _ => None,
        }
    }

    /// Get the QLoRA configuration if this is a QLoRA config
    pub fn qlora_config(&self) -> Option<&QLoRAConfig> {
        match &self.method_config {
            MethodConfig::QLoRA(config) => Some(config),
            _ => None,
        }
    }

    /// Get the AdaLoRA configuration if this is an AdaLoRA config
    pub fn adalora_config(&self) -> Option<&AdaLoRAConfig> {
        match &self.method_config {
            MethodConfig::AdaLoRA(config) => Some(config),
            _ => None,
        }
    }

    /// Get the IA³ configuration if this is an IA³ config
    pub fn ia3_config(&self) -> Option<&IA3Config> {
        match &self.method_config {
            MethodConfig::IA3(config) => Some(config),
            _ => None,
        }
    }

    /// Get the Prefix Tuning configuration if this is a Prefix Tuning config
    pub fn prefix_tuning_config(&self) -> Option<&PrefixTuningConfig> {
        match &self.method_config {
            MethodConfig::PrefixTuning(config) => Some(config),
            _ => None,
        }
    }

    /// Get the P-Tuning v2 configuration if this is a P-Tuning v2 config
    pub fn ptuning_v2_config(&self) -> Option<&PTuningV2Config> {
        match &self.method_config {
            MethodConfig::PTuningV2(config) => Some(config),
            _ => None,
        }
    }

    /// Get estimated parameter efficiency for this configuration
    pub fn estimated_efficiency(&self, base_params: usize) -> f64 {
        match &self.method_config {
            MethodConfig::LoRA(config) => {
                // Rough estimate: each adapted linear layer adds rank * (input_dim + output_dim) params
                // This is a simplification - actual efficiency depends on model architecture
                let typical_layer_size = 1000; // Rough estimate
                let adapter_params_per_layer =
                    config.rank * (typical_layer_size + typical_layer_size);
                let estimated_layers = self.target_modules.len().max(1);
                let total_adapter_params = adapter_params_per_layer * estimated_layers;

                total_adapter_params as f64 / (base_params + total_adapter_params) as f64
            }
            MethodConfig::QLoRA(config) => {
                // QLoRA: Same LoRA overhead + memory savings from quantization
                let lora_config = &config.lora_config;
                let typical_layer_size = 1000;
                let adapter_params_per_layer =
                    lora_config.rank * (typical_layer_size + typical_layer_size);
                let estimated_layers = self.target_modules.len().max(1);
                let total_adapter_params = adapter_params_per_layer * estimated_layers;

                // QLoRA has additional memory efficiency from quantization
                let quantization_savings = match config.bits {
                    4 => 0.25, // 4-bit uses ~25% of original memory
                    8 => 0.5,  // 8-bit uses ~50% of original memory
                    _ => 1.0,  // No savings for other bit widths
                };

                let effective_base_params = (base_params as f64 * quantization_savings) as usize;
                total_adapter_params as f64 / (effective_base_params + total_adapter_params) as f64
            }
            MethodConfig::AdaLoRA(config) => {
                // AdaLoRA: Adaptive rank allocation for better efficiency
                let typical_layer_size = 1000;
                // Use target rank for efficiency estimation since AdaLoRA adapts to this
                let effective_rank = config.target_rank;
                let adapter_params_per_layer =
                    effective_rank * (typical_layer_size + typical_layer_size);
                let estimated_layers = self.target_modules.len().max(1);
                let total_adapter_params = adapter_params_per_layer * estimated_layers;

                // AdaLoRA is more efficient due to adaptive rank allocation
                let adaptation_efficiency = 1.0 - config.budget_ratio; // Higher budget ratio = more aggressive adaptation
                let effective_adapter_params =
                    (total_adapter_params as f64 * adaptation_efficiency) as usize;

                effective_adapter_params as f64 / (base_params + effective_adapter_params) as f64
            }
            MethodConfig::IA3(config) => {
                // IA³: Extremely parameter-efficient (only scaling vectors)
                let typical_layer_size = 1000;
                let estimated_layers = self.target_modules.len().max(1);

                // IA³ only adds one parameter per dimension (scaling factor)
                let mut total_ia3_params = 0;
                if config.scale_attention {
                    total_ia3_params += typical_layer_size * estimated_layers / 2;
                    // Assume half layers have attention
                }
                if config.scale_feedforward {
                    total_ia3_params += typical_layer_size * estimated_layers; // All layers have feedforward
                }
                if config.scale_activations {
                    total_ia3_params += typical_layer_size * estimated_layers / 4;
                    // Some layers have activations
                }

                // IA³ is extremely efficient
                total_ia3_params as f64 / (base_params + total_ia3_params) as f64
            }
            MethodConfig::PrefixTuning(config) => {
                // Prefix tuning: prompt tokens per layer
                let params_per_layer = if config.reparameterize {
                    // Reparameterization MLP + prefix embeddings
                    config.prefix_length * config.reparameterization_dim * 2 + // prefix embeddings
                    config.reparameterization_dim * config.hidden_size * 4 // MLP params (linear layers + biases)
                } else {
                    // Direct prefix parameters
                    config.prefix_length * config.hidden_size * 2 // key + value
                };
                let total_prefix_params = params_per_layer * config.num_layers;

                total_prefix_params as f64 / (base_params + total_prefix_params) as f64
            }
            MethodConfig::PTuningV2(config) => {
                // P-Tuning v2: virtual tokens with optional projection
                let layers_with_prompts = match &config.prompt_layers {
                    super::ptuning_v2::PromptLayerConfig::All => config.num_layers,
                    super::ptuning_v2::PromptLayerConfig::Specific(layers) => layers.len(),
                    super::ptuning_v2::PromptLayerConfig::First(n) => *n.min(&config.num_layers),
                    super::ptuning_v2::PromptLayerConfig::Last(n) => *n.min(&config.num_layers),
                };

                let mut total_prompt_params =
                    config.num_virtual_tokens * config.token_dim * layers_with_prompts;

                // Add projection parameters if present
                if let Some(proj_dim) = config.prompt_projection_dim {
                    total_prompt_params += proj_dim * config.hidden_size * layers_with_prompts * 2;
                    // Linear + bias
                }

                total_prompt_params as f64 / (base_params + total_prompt_params) as f64
            }
            _ => 0.0,
        }
    }
}

impl Default for PEFTConfig {
    fn default() -> Self {
        Self::lora(LoRAConfig::default())
    }
}

/// Predefined PEFT configurations for common use cases
impl PEFTConfig {
    /// Configuration for fine-tuning large language models
    pub fn llm_lora() -> Self {
        Self::lora(LoRAConfig::for_llm()).with_target_modules(vec![
            "attention.query".to_string(),
            "attention.value".to_string(),
            "attention.key".to_string(),
            "attention.output".to_string(),
            "feed_forward.dense".to_string(),
        ])
    }

    /// Configuration for vision transformers
    pub fn vision_lora() -> Self {
        Self::lora(LoRAConfig::for_vision()).with_target_modules(vec![
            "attention.query".to_string(),
            "attention.value".to_string(),
            "mlp.dense".to_string(),
        ])
    }

    /// Configuration for efficient adaptation with minimal parameters
    pub fn efficient_lora() -> Self {
        Self::lora(LoRAConfig::for_efficiency()).with_target_modules(vec![
            "attention.query".to_string(),
            "attention.value".to_string(),
        ])
    }

    /// Configuration for research and experimentation
    pub fn research_lora() -> Self {
        Self::lora(LoRAConfig::for_high_rank())
            .with_freeze_base(false) // Allow base model training for research
            .with_save_only_adapter(false)
    }

    /// Configuration for memory-efficient fine-tuning of large language models
    pub fn llm_qlora() -> Self {
        Self::qlora(QLoRAConfig::for_llm()).with_target_modules(vec![
            "attention.query".to_string(),
            "attention.value".to_string(),
            "attention.key".to_string(),
            "attention.output".to_string(),
            "feed_forward.dense".to_string(),
        ])
    }

    /// Configuration for memory-efficient vision transformers
    pub fn vision_qlora() -> Self {
        Self::qlora(QLoRAConfig::for_vision()).with_target_modules(vec![
            "attention.query".to_string(),
            "attention.value".to_string(),
            "mlp.dense".to_string(),
        ])
    }

    /// Configuration for maximum memory efficiency with minimal parameters
    pub fn efficient_qlora() -> Self {
        Self::qlora(QLoRAConfig::for_efficiency()).with_target_modules(vec![
            "attention.query".to_string(),
            "attention.value".to_string(),
        ])
    }

    /// Configuration for adaptive fine-tuning of large language models
    pub fn llm_adalora() -> Self {
        Self::adalora(AdaLoRAConfig::for_llm()).with_target_modules(vec![
            "attention.query".to_string(),
            "attention.value".to_string(),
            "attention.key".to_string(),
            "attention.output".to_string(),
            "feed_forward.dense".to_string(),
        ])
    }

    /// Configuration for adaptive vision transformers
    pub fn vision_adalora() -> Self {
        Self::adalora(AdaLoRAConfig::for_vision()).with_target_modules(vec![
            "attention.query".to_string(),
            "attention.value".to_string(),
            "mlp.dense".to_string(),
        ])
    }

    /// Configuration for highly efficient adaptive fine-tuning
    pub fn efficient_adalora() -> Self {
        Self::adalora(AdaLoRAConfig::for_efficiency()).with_target_modules(vec![
            "attention.query".to_string(),
            "attention.value".to_string(),
        ])
    }

    /// Configuration for ultra-efficient attention scaling
    pub fn attention_ia3() -> Self {
        Self::ia3(IA3Config::for_language_attention()).with_target_modules(vec![
            "attention.query".to_string(),
            "attention.key".to_string(),
            "attention.value".to_string(),
        ])
    }

    /// Configuration for vision transformer scaling
    pub fn vision_ia3() -> Self {
        Self::ia3(IA3Config::for_vision()).with_target_modules(vec![
            "attention".to_string(),
            "feedforward".to_string(),
            "mlp".to_string(),
        ])
    }

    /// Configuration for maximum parameter efficiency
    pub fn efficient_ia3() -> Self {
        Self::ia3(IA3Config::for_efficiency()).with_target_modules(vec!["attention".to_string()])
    }

    /// Configuration for comprehensive IA³ adaptation
    pub fn comprehensive_ia3() -> Self {
        Self::ia3(IA3Config::for_comprehensive()).with_target_modules(vec![
            "attention".to_string(),
            "feedforward".to_string(),
            "output".to_string(),
        ])
    }

    /// Configuration for Prefix Tuning on language models
    pub fn llm_prefix_tuning() -> Self {
        Self::prefix_tuning(PrefixTuningConfig::for_generation(20, 768, 12, 12))
            .with_target_modules(vec!["attention".to_string(), "transformer".to_string()])
    }

    /// Configuration for Prefix Tuning for understanding tasks
    pub fn understanding_prefix_tuning() -> Self {
        Self::prefix_tuning(PrefixTuningConfig::for_understanding(15, 768, 12, 12))
            .with_target_modules(vec!["attention".to_string()])
    }

    /// Configuration for P-Tuning v2 on language models
    pub fn llm_ptuning_v2() -> Self {
        Self::ptuning_v2(PTuningV2Config::for_nlu(100, 768, 12)).with_target_modules(vec![
            "transformer".to_string(),
            "layer".to_string(),
            "attention".to_string(),
        ])
    }

    /// Configuration for P-Tuning v2 for generation tasks
    pub fn generation_ptuning_v2() -> Self {
        Self::ptuning_v2(PTuningV2Config::for_nlg(150, 768, 12))
            .with_target_modules(vec!["transformer".to_string(), "layer".to_string()])
    }

    /// Configuration for P-Tuning v2 for conditional generation
    pub fn conditional_ptuning_v2() -> Self {
        Self::ptuning_v2(PTuningV2Config::for_conditional_generation(120, 1024, 24))
            .with_target_modules(vec![
                "encoder".to_string(),
                "decoder".to_string(),
                "transformer".to_string(),
            ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peft_method_properties() {
        let lora = PEFTMethod::LoRA;
        assert_eq!(lora.name(), "LoRA");
        assert!(lora.is_implemented());
        assert!(lora.description().contains("Low-rank"));

        let qlora = PEFTMethod::QLoRA;
        assert!(qlora.is_implemented());
    }

    #[test]
    fn test_module_targeting() {
        let config = PEFTConfig::llm_lora();

        assert!(config.should_adapt_module("attention.query"));
        assert!(config.should_adapt_module("model.layers.0.attention.query"));
        assert!(!config.should_adapt_module("embedding"));
        assert!(!config.should_adapt_module("layernorm"));
    }

    #[test]
    fn test_module_exclusion() {
        let config = PEFTConfig::default()
            .with_target_modules(vec!["dense".to_string()])
            .with_exclude_modules(vec!["embedding".to_string()]);

        assert!(config.should_adapt_module("linear.dense"));
        assert!(!config.should_adapt_module("embedding.dense"));
    }

    #[test]
    fn test_predefined_configs() {
        let llm_config = PEFTConfig::llm_lora();
        assert!(llm_config
            .target_modules
            .contains(&"attention.query".to_string()));
        assert!(llm_config.freeze_base);

        let research_config = PEFTConfig::research_lora();
        assert!(!research_config.freeze_base);
        assert!(!research_config.save_only_adapter);
    }

    #[test]
    fn test_efficiency_estimation() {
        let config = PEFTConfig::efficient_lora();
        let base_params = 1_000_000;
        let efficiency = config.estimated_efficiency(base_params);

        // Should be a small fraction for efficient config
        assert!(efficiency > 0.0 && efficiency < 0.1);
    }

    #[test]
    fn test_qlora_config_creation() {
        let qlora_config = PEFTConfig::llm_qlora();
        assert_eq!(qlora_config.method, PEFTMethod::QLoRA);
        assert!(qlora_config.qlora_config().is_some());
        assert!(qlora_config
            .target_modules
            .contains(&"attention.query".to_string()));
        assert!(qlora_config.freeze_base);
    }

    #[test]
    fn test_qlora_efficiency_estimation() {
        let qlora_config = PEFTConfig::efficient_qlora();
        let base_params = 1_000_000;
        let efficiency = qlora_config.estimated_efficiency(base_params);

        // QLoRA should be more efficient than regular LoRA due to quantization
        let lora_config = PEFTConfig::efficient_lora();
        let lora_efficiency = lora_config.estimated_efficiency(base_params);

        assert!(
            efficiency > lora_efficiency,
            "QLoRA should be more efficient than LoRA"
        );
        assert!(efficiency > 0.0 && efficiency < 0.5);
    }

    #[test]
    fn test_qlora_vs_lora_config_access() {
        let qlora_config = PEFTConfig::efficient_qlora();

        // Should be able to access both QLoRA config and underlying LoRA config
        assert!(qlora_config.qlora_config().is_some());
        assert!(qlora_config.lora_config().is_some());

        let lora_config = PEFTConfig::efficient_lora();
        assert!(lora_config.lora_config().is_some());
        assert!(lora_config.qlora_config().is_none());
        assert!(lora_config.adalora_config().is_none());
    }

    #[test]
    fn test_adalora_config_creation() {
        let adalora_config = PEFTConfig::llm_adalora();
        assert_eq!(adalora_config.method, PEFTMethod::AdaLoRA);
        assert!(adalora_config.adalora_config().is_some());
        assert!(adalora_config
            .target_modules
            .contains(&"attention.query".to_string()));
        assert!(adalora_config.freeze_base);
    }

    #[test]
    fn test_adalora_efficiency_estimation() {
        let adalora_config = PEFTConfig::efficient_adalora();
        let base_params = 1_000_000;
        let efficiency = adalora_config.estimated_efficiency(base_params);

        // AdaLoRA should be more efficient than regular LoRA due to adaptive rank allocation
        let lora_config = PEFTConfig::efficient_lora();
        let lora_efficiency = lora_config.estimated_efficiency(base_params);

        assert!(
            efficiency <= lora_efficiency,
            "AdaLoRA should be more efficient than LoRA"
        );
        assert!(efficiency > 0.0 && efficiency < 0.3);
    }

    #[test]
    fn test_adalora_vs_other_configs() {
        let adalora_config = PEFTConfig::efficient_adalora();

        // Should be able to access AdaLoRA config but not others
        assert!(adalora_config.adalora_config().is_some());
        assert!(adalora_config.lora_config().is_none()); // AdaLoRA doesn't expose inner LoRA config this way
        assert!(adalora_config.qlora_config().is_none());

        // Check config properties
        let config = adalora_config.adalora_config().unwrap();
        assert!(config.target_rank <= config.init_rank);
        assert!(config.budget_ratio > 0.0 && config.budget_ratio < 1.0);
    }

    #[test]
    fn test_ia3_config_creation() {
        let ia3_config = PEFTConfig::attention_ia3();
        assert_eq!(ia3_config.method, PEFTMethod::IA3);
        assert!(ia3_config.ia3_config().is_some());
        assert!(ia3_config
            .target_modules
            .contains(&"attention.query".to_string()));
        assert!(ia3_config.freeze_base);
    }

    #[test]
    fn test_ia3_efficiency_estimation() {
        let ia3_config = PEFTConfig::efficient_ia3();
        let base_params = 1_000_000;
        let efficiency = ia3_config.estimated_efficiency(base_params);

        // IA³ should be the most efficient method
        let lora_config = PEFTConfig::efficient_lora();
        let lora_efficiency = lora_config.estimated_efficiency(base_params);
        let adalora_config = PEFTConfig::efficient_adalora();
        let adalora_efficiency = adalora_config.estimated_efficiency(base_params);

        assert!(
            efficiency < lora_efficiency,
            "IA³ should be more efficient than LoRA"
        );
        assert!(
            efficiency < adalora_efficiency,
            "IA³ should be more efficient than AdaLoRA"
        );
        assert!(efficiency > 0.0 && efficiency < 0.1);
    }

    #[test]
    fn test_ia3_vs_other_configs() {
        let ia3_config = PEFTConfig::efficient_ia3();

        // Should be able to access IA³ config but not others
        assert!(ia3_config.ia3_config().is_some());
        assert!(ia3_config.lora_config().is_none());
        assert!(ia3_config.qlora_config().is_none());
        assert!(ia3_config.adalora_config().is_none());

        // Check config properties
        let config = ia3_config.ia3_config().unwrap();
        assert!(config.scale_attention);
        assert!(config.learning_rate_multiplier > 1.0); // IA³ typically needs higher LR
    }

    #[test]
    fn test_all_peft_methods_implemented() {
        // Test that all main PEFT methods are now implemented
        assert!(PEFTMethod::LoRA.is_implemented());
        assert!(PEFTMethod::QLoRA.is_implemented());
        assert!(PEFTMethod::AdaLoRA.is_implemented());
        assert!(PEFTMethod::IA3.is_implemented());
        assert!(PEFTMethod::PrefixTuning.is_implemented());
        assert!(PEFTMethod::PTuningV2.is_implemented());

        // Custom methods should also be implemented
        assert!(PEFTMethod::Custom("test".to_string()).is_implemented());
    }

    #[test]
    fn test_efficiency_comparison() {
        let base_params = 10_000_000; // 10M parameter model

        let lora_efficiency = PEFTConfig::efficient_lora().estimated_efficiency(base_params);
        let qlora_efficiency = PEFTConfig::efficient_qlora().estimated_efficiency(base_params);
        let adalora_efficiency = PEFTConfig::efficient_adalora().estimated_efficiency(base_params);
        let ia3_efficiency = PEFTConfig::efficient_ia3().estimated_efficiency(base_params);

        // IA³ should be most efficient, QLoRA should be more efficient than LoRA
        assert!(ia3_efficiency < adalora_efficiency);
        assert!(adalora_efficiency <= lora_efficiency);
        assert!(qlora_efficiency > lora_efficiency); // QLoRA has quantization savings

        // All should be reasonably efficient
        assert!(lora_efficiency < 0.3);
        assert!(qlora_efficiency < 0.5);
        assert!(adalora_efficiency < 0.2);
        assert!(ia3_efficiency < 0.1);
    }
}
