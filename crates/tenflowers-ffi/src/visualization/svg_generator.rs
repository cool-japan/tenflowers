//! SVG diagram generator for gradient flow visualization
//!
//! This module provides comprehensive SVG diagram generation for gradient flow analysis.

use super::gradient_flow::PyGradientFlowAnalysis;
use scirs2_core::numeric::ScientificNumber;

/// Generate an SVG diagram for gradient flow visualization
pub fn generate_svg_diagram(analysis: &PyGradientFlowAnalysis) -> String {
    let stats = &analysis.inner.flow_statistics;
    let health_score = analysis.get_health_score();

    let width = 800;
    let height = 600;
    let margin = 80;

    let min_grad = stats.min_gradient_magnitude.to_f64().unwrap_or(0.0);
    let max_grad = stats.max_gradient_magnitude.to_f64().unwrap_or(0.0);
    let vanishing_pct = stats.vanishing_percentage.to_f64().unwrap_or(0.0);

    // Generate color scheme based on health score
    let (primary_color, secondary_color) = if health_score >= 80.0 {
        ("#4caf50", "#66bb6a") // Green
    } else if health_score >= 60.0 {
        ("#2196F3", "#42a5f5") // Blue
    } else if health_score >= 40.0 {
        ("#ff9800", "#ffa726") // Orange
    } else {
        ("#f44336", "#ef5350") // Red
    };

    let font_family = "Arial, sans-serif";

    // Build SVG header
    let mut svg = String::new();
    svg.push_str(&format!("<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {} {}\" width=\"{}\" height=\"{}\">\n", width, height, width, height));

    // Definitions
    svg.push_str("    <defs>\n");
    svg.push_str("        <linearGradient id=\"healthGradient\" x1=\"0%\" y1=\"0%\" x2=\"100%\" y2=\"0%\">\n");
    svg.push_str(&format!(
        "            <stop offset=\"0%\" style=\"stop-color:{};stop-opacity:1\" />\n",
        primary_color
    ));
    svg.push_str(&format!(
        "            <stop offset=\"100%\" style=\"stop-color:{};stop-opacity:1\" />\n",
        secondary_color
    ));
    svg.push_str("        </linearGradient>\n");
    svg.push_str("        <linearGradient id=\"flowGradient\" x1=\"0%\" y1=\"0%\" x2=\"100%\" y2=\"100%\">\n");
    svg.push_str(
        "            <stop offset=\"0%\" style=\"stop-color:#667eea;stop-opacity:1\" />\n",
    );
    svg.push_str(
        "            <stop offset=\"100%\" style=\"stop-color:#764ba2;stop-opacity:1\" />\n",
    );
    svg.push_str("        </linearGradient>\n");
    svg.push_str("        <filter id=\"shadow\">\n");
    svg.push_str(
        "            <feDropShadow dx=\"0\" dy=\"2\" stdDeviation=\"3\" flood-opacity=\"0.3\"/>\n",
    );
    svg.push_str("        </filter>\n");
    svg.push_str("    </defs>\n\n");

    // Background
    svg.push_str(&format!(
        "    <rect width=\"{}\" height=\"{}\" fill=\"#f8f9fa\" />\n\n",
        width, height
    ));

    // Title section
    svg.push_str(&format!(
        "    <rect x=\"0\" y=\"0\" width=\"{}\" height=\"80\" fill=\"url(#flowGradient)\" />\n",
        width
    ));
    svg.push_str(&format!("    <text x=\"{}\" y=\"35\" font-family=\"{}\" font-size=\"28\" font-weight=\"bold\" fill=\"white\" text-anchor=\"middle\">\n", width / 2, font_family));
    svg.push_str("        🔬 Gradient Flow Analysis\n");
    svg.push_str("    </text>\n");
    svg.push_str(&format!("    <text x=\"{}\" y=\"60\" font-family=\"{}\" font-size=\"14\" fill=\"white\" text-anchor=\"middle\" opacity=\"0.95\">\n", width / 2, font_family));
    svg.push_str("        Neural Network Gradient Flow Visualization\n");
    svg.push_str("    </text>\n\n");

    // Health score section
    svg.push_str("    <g id=\"health-score\">\n");
    svg.push_str("        <rect x=\"20\" y=\"100\" width=\"760\" height=\"80\" rx=\"12\" fill=\"white\" filter=\"url(#shadow)\" />\n");
    svg.push_str(&format!("        <text x=\"40\" y=\"130\" font-family=\"{}\" font-size=\"16\" font-weight=\"bold\" fill=\"#333\">\n", font_family));
    svg.push_str("            Overall Health Score\n");
    svg.push_str("        </text>\n");
    svg.push_str(&format!("        <text x=\"740\" y=\"145\" font-family=\"{}\" font-size=\"36\" font-weight=\"bold\" fill=\"url(#healthGradient)\" text-anchor=\"end\">\n", font_family));
    svg.push_str(&format!("            {:.1}%\n", health_score));
    svg.push_str("        </text>\n");

    // Progress bar
    svg.push_str("        <rect x=\"40\" y=\"155\" width=\"700\" height=\"15\" rx=\"7.5\" fill=\"#e0e0e0\" />\n");
    let progress_width = (health_score * 7.0) as i32;
    svg.push_str(&format!("        <rect x=\"40\" y=\"155\" width=\"{}\" height=\"15\" rx=\"7.5\" fill=\"url(#healthGradient)\" />\n", progress_width));
    svg.push_str("    </g>\n\n");

    // Statistics grid
    svg.push_str("    <g id=\"statistics-grid\">\n");
    svg.push_str(&generate_statistics_grid(stats, margin, font_family));
    svg.push_str("    </g>\n\n");

    // Network visualization placeholder
    svg.push_str("    <g id=\"network-viz\">\n");
    svg.push_str(&generate_network_visualization(stats, margin, font_family));
    svg.push_str("    </g>\n\n");

    // Legend
    svg.push_str(&format!(
        "    <g id=\"legend\" transform=\"translate(20, {})\">\n",
        height - 120
    ));
    svg.push_str("        <rect x=\"0\" y=\"0\" width=\"760\" height=\"70\" rx=\"8\" fill=\"white\" filter=\"url(#shadow)\" />\n");
    svg.push_str(&format!("        <text x=\"20\" y=\"25\" font-family=\"{}\" font-size=\"14\" font-weight=\"bold\" fill=\"#333\">Legend</text>\n", font_family));
    svg.push_str("        <circle cx=\"40\" cy=\"45\" r=\"8\" fill=\"url(#healthGradient)\" />\n");
    svg.push_str(&format!("        <text x=\"55\" y=\"50\" font-family=\"{}\" font-size=\"12\" fill=\"#666\">Healthy Gradient</text>\n", font_family));
    svg.push_str("        <circle cx=\"200\" cy=\"45\" r=\"8\" fill=\"#ff9800\" />\n");
    svg.push_str(&format!("        <text x=\"215\" y=\"50\" font-family=\"{}\" font-size=\"12\" fill=\"#666\">Warning</text>\n", font_family));
    svg.push_str("        <circle cx=\"320\" cy=\"45\" r=\"8\" fill=\"#f44336\" />\n");
    svg.push_str(&format!("        <text x=\"335\" y=\"50\" font-family=\"{}\" font-size=\"12\" fill=\"#666\">Critical</text>\n", font_family));
    svg.push_str("    </g>\n\n");

    // Footer
    svg.push_str(&format!("    <text x=\"{}\" y=\"{}\" font-family=\"{}\" font-size=\"11\" fill=\"#999\" text-anchor=\"middle\">\n", width / 2, height - 20, font_family));
    svg.push_str(&format!(
        "        Generated by TenFlowers FFI • {} UTC\n",
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S")
    ));
    svg.push_str("    </text>\n");

    svg.push_str("</svg>");
    svg
}

fn generate_statistics_grid(
    stats: &tenflowers_autograd::gradient_visualization::FlowStatistics<f32>,
    margin: usize,
    font_family: &str,
) -> String {
    let y_start = 200;
    let card_width = 140;
    let card_height = 90;
    let gap = 15;

    let cards = [
        ("Nodes", stats.total_nodes.to_string(), "📊"),
        ("Edges", stats.total_edges.to_string(), "🔗"),
        (
            "Max Grad",
            format!(
                "{:.2e}",
                stats.max_gradient_magnitude.to_f64().unwrap_or(0.0)
            ),
            "⬆️",
        ),
        (
            "Min Grad",
            format!(
                "{:.2e}",
                stats.min_gradient_magnitude.to_f64().unwrap_or(0.0)
            ),
            "⬇️",
        ),
        (
            "Vanishing",
            format!("{:.1}%", stats.vanishing_percentage.to_f64().unwrap_or(0.0)),
            "⚠️",
        ),
    ];

    let mut grid = String::new();
    for (i, (title, value, icon)) in cards.iter().enumerate() {
        let x = margin + i * (card_width + gap);
        grid.push_str(&format!(
            "        <g transform=\"translate({}, {})\">\n",
            x, y_start
        ));
        grid.push_str(&format!("            <rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" rx=\"8\" fill=\"white\" filter=\"url(#shadow)\" />\n", card_width, card_height));
        grid.push_str(&format!(
            "            <text x=\"15\" y=\"28\" font-size=\"24\">{}</text>\n",
            icon
        ));
        grid.push_str(&format!("            <text x=\"15\" y=\"50\" font-family=\"{}\" font-size=\"11\" fill=\"#666\">{}</text>\n", font_family, title));
        grid.push_str(&format!("            <text x=\"15\" y=\"75\" font-family=\"{}\" font-size=\"16\" font-weight=\"bold\" fill=\"#333\">{}</text>\n", font_family, value));
        grid.push_str("        </g>\n");
    }
    grid
}

fn generate_network_visualization(
    stats: &tenflowers_autograd::gradient_visualization::FlowStatistics<f32>,
    margin: usize,
    font_family: &str,
) -> String {
    let y_center = 380;
    let node_radius = 25;
    let total_nodes = stats.total_nodes.clamp(3, 8);
    let spacing = 90;

    let mut viz = String::new();

    // Draw nodes
    for i in 0..total_nodes {
        let x = margin + 50 + i * spacing;
        let color = if i == 0 || i == total_nodes - 1 {
            "#4caf50" // Input/output nodes
        } else {
            "#2196F3" // Hidden nodes
        };

        viz.push_str(&format!("        <circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"{}\" opacity=\"0.9\" filter=\"url(#shadow)\" />\n", x, y_center, node_radius, color));
        viz.push_str(&format!("        <circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"none\" stroke=\"white\" stroke-width=\"2\" />\n", x, y_center, node_radius - 3));
        viz.push_str(&format!("        <text x=\"{}\" y=\"{}\" font-family=\"{}\" font-size=\"12\" font-weight=\"bold\" fill=\"white\" text-anchor=\"middle\">\n", x, y_center + 4, font_family));
        viz.push_str(&format!("            L{}\n", i));
        viz.push_str("        </text>\n");

        // Draw edges
        if i > 0 {
            let x1 = margin + 50 + (i - 1) * spacing + node_radius;
            let x2 = x - node_radius;
            viz.push_str(&format!("        <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#667eea\" stroke-width=\"2\" opacity=\"0.6\" />\n", x1, y_center, x2, y_center));
        }
    }

    viz
}
