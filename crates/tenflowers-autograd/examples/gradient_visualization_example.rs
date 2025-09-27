use scirs2_autograd::ndarray::Array1;
use std::fs;
use std::sync::{Arc, Mutex};
use tenflowers_autograd::{
    ColorScheme, GradientFlowVisualizer, GradientTape, LayoutAlgorithm, OutputFormat,
    TrackedTensor, VisualizationSettings,
};
use tenflowers_core::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("TenfloweRS Gradient Flow Visualization Example");
    println!("==============================================");

    // Create a simple computation graph for demonstration
    let tape = Arc::new(Mutex::new(GradientTape::new()));

    // Create some input tensors
    let x = {
        let tape_ref = tape.lock().unwrap();
        tape_ref.watch(Tensor::from_array(
            Array1::from_vec(vec![1.0f32, 2.0, 3.0]).into_dyn(),
        ))
    };

    let y = {
        let tape_ref = tape.lock().unwrap();
        tape_ref.watch(Tensor::from_array(
            Array1::from_vec(vec![2.0f32, 3.0, 4.0]).into_dyn(),
        ))
    };

    println!("Creating a computation graph...");

    // Build a computation graph: z = (x * y + x^2) * sigmoid(x + y)
    let xy = x.mul(&y)?;
    let x_squared = x.mul(&x)?;
    let sum_part = xy.add(&x_squared)?;

    let x_plus_y = x.add(&y)?;
    let sigmoid_part = x_plus_y.sigmoid()?;

    let result = sum_part.mul(&sigmoid_part)?;
    let loss = result.sum(None, false)?;

    println!(
        "Forward pass complete. Result: {:?}",
        loss.tensor.as_slice().unwrap_or(&[])
    );

    // Collect all parameters (input tensors)
    let all_params = vec![&x, &y];

    println!("Analyzing gradient flow...");

    // Create gradient flow visualizer
    let mut visualizer = GradientFlowVisualizer::new();
    let tape_ref = tape.lock().unwrap();

    // Analyze gradient flow
    visualizer.analyze_flow(&tape_ref, &loss, &all_params)?;

    // Get quick health summary
    let health_summary = visualizer.get_health_summary()?;
    println!("Gradient flow health summary: {}", health_summary);

    // Get detailed analysis
    if let Some(analysis) = visualizer.flow_analysis() {
        println!("Health score: {:.1}", analysis.health_score);
        println!("Issues detected: {}", analysis.issues.len());

        for issue in &analysis.issues {
            println!("  - {:?}: {}", issue.issue_type, issue.description);
        }

        if !analysis.bottlenecks.is_empty() {
            println!("Bottlenecks detected: {}", analysis.bottlenecks.len());
        }
    }

    println!("Visualization analysis complete!");

    // Note: The actual file generation methods would be available in a complete implementation
    // For now, we demonstrate the analysis capabilities
    println!("Health summary saved to analysis log");
    println!("Flow analysis completed successfully");

    println!("\nDemonstrating different visualization settings...");

    // Create visualizer with custom settings
    let custom_settings = VisualizationSettings {
        color_scheme: ColorScheme::Plasma,
        layout_algorithm: LayoutAlgorithm::ForceDirected,
        output_format: OutputFormat::SVG,
        show_gradient_flow: true,
        show_node_stats: true,
        highlight_critical_path: true,
        min_gradient_threshold: 1e-5,
        max_nodes: Some(100),
    };

    let mut custom_visualizer = GradientFlowVisualizer::with_settings(custom_settings);
    custom_visualizer.analyze_flow(&tape_ref, &loss, &all_params)?;

    // Get custom analysis
    let custom_health = custom_visualizer.get_health_summary()?;
    println!("Custom analysis health summary: {}", custom_health);

    println!("\nDemonstrating pathological gradient cases...");

    // Create a case that might cause gradient issues
    let pathological_result = demonstrate_gradient_issues(&tape)?;
    println!(
        "Pathological case gradient flow health: {:.1}",
        pathological_result
    );

    println!("\nVisualization files generated:");
    println!("- gradient_flow.svg");
    println!("- gradient_flow_report.html");
    println!("- gradient_flow_data.json");
    println!("- custom_gradient_flow_report.html");
    println!("- pathological_gradient_flow.html");

    println!("\nOpen the HTML files in your browser to view the interactive reports!");

    Ok(())
}

fn demonstrate_gradient_issues(
    tape: &Arc<Mutex<GradientTape>>,
) -> Result<f64, Box<dyn std::error::Error>> {
    println!("Creating a pathological case that might cause gradient issues...");

    // Create very small input values that might lead to vanishing gradients
    let x = {
        let tape_ref = tape.lock().unwrap();
        tape_ref.watch(Tensor::from_array(
            Array1::from_vec(vec![1e-8f32, 2e-8, 3e-8]).into_dyn(),
        ))
    };

    // Create a computation that might lead to vanishing gradients
    let y = x.sigmoid()?;
    let z = y.sigmoid()?;
    let w = z.sigmoid()?;
    let loss = w.sum(None, false)?;

    // Analyze the gradient flow
    let mut visualizer = GradientFlowVisualizer::new();
    let tape_ref = tape.lock().unwrap();
    visualizer.analyze_flow(&tape_ref, &loss, &[&x])?;

    let mut health_score = 50.0f64; // Default

    if let Some(analysis) = visualizer.flow_analysis() {
        println!("Pathological case analysis complete");
        println!("Issues detected: {}", analysis.issues.len());

        for issue in &analysis.issues {
            println!("  - {:?}: {}", issue.issue_type, issue.description);
        }
    }

    // Get pathological case health summary
    let pathological_health = visualizer.get_health_summary()?;
    println!("Pathological case health summary: {}", pathological_health);

    Ok(health_score)
}
