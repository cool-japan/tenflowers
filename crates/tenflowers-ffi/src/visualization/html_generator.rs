//! HTML report generator for gradient flow visualization
//!
//! This module provides comprehensive HTML report generation for gradient flow analysis.

use super::gradient_flow::PyGradientFlowAnalysis;
use scirs2_core::numeric::ScientificNumber;

/// Generate a comprehensive HTML report for gradient flow analysis
pub fn generate_html_report(analysis: &PyGradientFlowAnalysis) -> String {
    let stats = &analysis.inner.flow_statistics;
    let health_score = analysis.get_health_score();
    let recommendations = analysis.get_recommendations();
    let issues = analysis.get_issues();
    let has_issues = analysis.has_issues();

    let min_grad = stats.min_gradient_magnitude.to_f64().unwrap_or(0.0);
    let max_grad = stats.max_gradient_magnitude.to_f64().unwrap_or(0.0);
    let vanishing_pct = stats.vanishing_percentage.to_f64().unwrap_or(0.0);

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TenFlowers Gradient Flow Analysis Report</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 50px 40px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 3em;
            margin-bottom: 15px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            animation: fadeInDown 0.6s ease;
        }}
        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.95;
            animation: fadeInUp 0.8s ease;
        }}
        .content {{
            padding: 40px;
        }}
        .health-score {{
            text-align: center;
            padding: 50px 40px;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            border-radius: 12px;
            margin-bottom: 40px;
            box-shadow: 0 10px 30px rgba(245, 87, 108, 0.3);
            animation: scaleIn 0.5s ease;
        }}
        .health-score h2 {{
            font-size: 1.8em;
            margin-bottom: 20px;
        }}
        .health-value {{
            font-size: 5em;
            font-weight: bold;
            margin: 20px 0;
            text-shadow: 3px 3px 6px rgba(0,0,0,0.2);
        }}
        .progress-bar {{
            width: 100%;
            max-width: 600px;
            height: 40px;
            background: rgba(255,255,255,0.3);
            border-radius: 20px;
            overflow: hidden;
            margin: 20px auto;
            position: relative;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #4caf50, #8bc34a);
            border-radius: 20px;
            transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 1.1em;
            box-shadow: 0 4px 12px rgba(76, 175, 80, 0.4);
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 25px;
            margin: 40px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 30px;
            border-radius: 12px;
            border-left: 5px solid #667eea;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .metric-card:hover {{
            transform: translateY(-8px) scale(1.02);
            box-shadow: 0 12px 24px rgba(0,0,0,0.15);
        }}
        .metric-icon {{
            font-size: 2em;
            margin-bottom: 10px;
        }}
        .metric-title {{
            font-size: 0.9em;
            color: #555;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 12px;
            font-weight: 600;
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #333;
        }}
        .section {{
            margin: 40px 0;
            padding: 30px;
            border-radius: 12px;
            border-left: 5px solid;
            animation: fadeIn 0.6s ease;
        }}
        .section-title {{
            font-size: 1.8em;
            color: #333;
            margin-bottom: 25px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .recommendations {{
            background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%);
            border-color: #4caf50;
        }}
        .issues {{
            background: linear-gradient(135deg, #ffebee 0%, #f8d7da 100%);
            border-color: #f44336;
        }}
        .insights {{
            background: linear-gradient(135deg, #e3f2fd 0%, #cfe8fc 100%);
            border-color: #2196F3;
        }}
        .success-section {{
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            border-color: #28a745;
        }}
        .list-item {{
            background: white;
            margin: 12px 0;
            padding: 18px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            display: flex;
            align-items: center;
            transition: all 0.2s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .list-item:hover {{
            transform: translateX(5px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .status-indicator {{
            width: 14px;
            height: 14px;
            border-radius: 50%;
            margin-right: 15px;
            flex-shrink: 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        .status-good {{ background: linear-gradient(135deg, #4caf50, #66bb6a); }}
        .status-warning {{ background: linear-gradient(135deg, #ff9800, #ffa726); }}
        .status-error {{ background: linear-gradient(135deg, #f44336, #ef5350); }}
        .badge {{
            display: inline-block;
            padding: 6px 14px;
            border-radius: 14px;
            font-size: 0.85em;
            font-weight: 700;
            margin-left: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .badge-success {{ background: #4caf50; color: white; box-shadow: 0 2px 4px rgba(76, 175, 80, 0.3); }}
        .badge-warning {{ background: #ff9800; color: white; box-shadow: 0 2px 4px rgba(255, 152, 0, 0.3); }}
        .badge-danger {{ background: #f44336; color: white; box-shadow: 0 2px 4px rgba(244, 67, 54, 0.3); }}
        .footer {{
            text-align: center;
            padding: 40px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            color: #666;
            font-size: 0.95em;
        }}
        .footer strong {{
            color: #333;
            font-size: 1.1em;
        }}
        .footer a {{
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
            transition: color 0.2s ease;
        }}
        .footer a:hover {{
            color: #764ba2;
            text-decoration: underline;
        }}
        @keyframes fadeInDown {{
            from {{ opacity: 0; transform: translateY(-20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        @keyframes scaleIn {{
            from {{ opacity: 0; transform: scale(0.9); }}
            to {{ opacity: 1; transform: scale(1); }}
        }}
        .timestamp {{
            opacity: 0.9;
            margin-top: 8px;
            font-size: 0.95em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 TenFlowers Gradient Flow Analysis</h1>
            <div class="subtitle">Comprehensive Neural Network Gradient Flow Report</div>
            <div class="timestamp">Generated: {}</div>
        </div>

        <div class="content">
            <div class="health-score">
                <h2>🎯 Overall Health Score</h2>
                <div class="health-value">{:.1}%</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {:.1}%">
                        {:.1}%
                    </div>
                </div>
                <p style="font-size: 1.3em; margin-top: 20px; font-weight: 500;">{}</p>
            </div>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-icon">📊</div>
                    <div class="metric-title">Total Nodes</div>
                    <div class="metric-value">{}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">🔗</div>
                    <div class="metric-title">Total Edges</div>
                    <div class="metric-value">{}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">⬆️</div>
                    <div class="metric-title">Max Gradient</div>
                    <div class="metric-value">{:.2e}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">⬇️</div>
                    <div class="metric-title">Min Gradient</div>
                    <div class="metric-value">{:.2e}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">⚠️</div>
                    <div class="metric-title">Vanishing %</div>
                    <div class="metric-value">{:.1}%</div>
                </div>
            </div>

            {}

            <div class="section insights">
                <div class="section-title">📊 Analysis Insights</div>
                <div class="list-item">
                    <span class="status-indicator {}"></span>
                    <div>
                        <strong>Gradient Range:</strong>&nbsp; {:.2e} to {:.2e}
                        <span class="badge {}">{}</span>
                    </div>
                </div>
                <div class="list-item">
                    <span class="status-indicator {}"></span>
                    <div>
                        <strong>Vanishing Risk:</strong>&nbsp; {:.1}%
                        <span class="badge {}">{}</span>
                    </div>
                </div>
                <div class="list-item">
                    <span class="status-indicator {}"></span>
                    <div>
                        <strong>Network Complexity:</strong>&nbsp; {} nodes with {} connections
                    </div>
                </div>
            </div>

            <div class="section recommendations">
                <div class="section-title">💡 Recommendations</div>
                {}
            </div>
        </div>

        <div class="footer">
            <p><strong>Generated by TenFlowers FFI Gradient Flow Visualizer v0.1.0</strong></p>
            <p style="margin-top: 12px;">
                For more information, visit
                <a href="https://github.com/cool-japan/tenflowers" target="_blank">TenFlowers Documentation</a>
            </p>
            <p style="margin-top: 8px; font-size: 0.85em; opacity: 0.8;">
                Built with Rust 🦀 + PyO3 🐍
            </p>
        </div>
    </div>

    <script>
        // Animate progress bar on load
        window.addEventListener('load', function() {{
            const progressBar = document.querySelector('.progress-fill');
            if (progressBar) {{
                setTimeout(() => {{
                    progressBar.style.width = '{:.1}%';
                }}, 100);
            }}
        }});
    </script>
</body>
</html>"#,
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
        health_score,
        health_score,
        health_score,
        get_health_message(health_score),
        stats.total_nodes,
        stats.total_edges,
        max_grad,
        min_grad,
        vanishing_pct,
        // Issues or success section
        if has_issues {
            format!(
                r#"<div class="section issues">
                <div class="section-title">⚠️ Detected Issues</div>
                {}
            </div>"#,
                issues
                    .iter()
                    .map(|issue| format!(
                        r#"<div class="list-item">
                        <span class="status-indicator status-error"></span>
                        <span>{}</span>
                    </div>"#,
                        issue
                    ))
                    .collect::<Vec<_>>()
                    .join("")
            )
        } else {
            r#"<div class="section success-section">
                <div class="section-title">✅ Status</div>
                <div class="list-item">
                    <span class="status-indicator status-good"></span>
                    <span><strong>No critical issues detected</strong> - Gradient flow is healthy</span>
                </div>
            </div>"#.to_string()
        },
        // Gradient range indicator
        if max_grad > 10.0 {
            "status-error"
        } else if max_grad > 1.0 {
            "status-warning"
        } else {
            "status-good"
        },
        min_grad,
        max_grad,
        if max_grad > 10.0 {
            "badge-danger"
        } else if max_grad > 1.0 {
            "badge-warning"
        } else {
            "badge-success"
        },
        if max_grad > 10.0 {
            "High"
        } else if max_grad > 1.0 {
            "Elevated"
        } else {
            "Normal"
        },
        // Vanishing risk
        if vanishing_pct > 20.0 {
            "status-error"
        } else if vanishing_pct > 10.0 {
            "status-warning"
        } else {
            "status-good"
        },
        vanishing_pct,
        if vanishing_pct > 20.0 {
            "badge-danger"
        } else if vanishing_pct > 10.0 {
            "badge-warning"
        } else {
            "badge-success"
        },
        if vanishing_pct > 20.0 {
            "High"
        } else if vanishing_pct > 10.0 {
            "Medium"
        } else {
            "Low"
        },
        // Network complexity
        if stats.total_nodes > 100 {
            "status-warning"
        } else {
            "status-good"
        },
        stats.total_nodes,
        stats.total_edges,
        // Recommendations
        recommendations
            .iter()
            .enumerate()
            .map(|(i, rec)| format!(
                r#"<div class="list-item">
                <span class="status-indicator status-good"></span>
                <span><strong>{}.</strong> {}</span>
            </div>"#,
                i + 1,
                rec
            ))
            .collect::<Vec<_>>()
            .join(""),
        health_score
    )
}

fn get_health_message(health_score: f64) -> &'static str {
    if health_score >= 90.0 {
        "✨ Outstanding! Gradient flow is optimal"
    } else if health_score >= 80.0 {
        "✅ Excellent gradient flow health"
    } else if health_score >= 70.0 {
        "✓ Good gradient flow with minor room for improvement"
    } else if health_score >= 60.0 {
        "⚠ Adequate gradient flow - monitor closely"
    } else if health_score >= 50.0 {
        "⚠ Moderate gradient flow - attention recommended"
    } else if health_score >= 40.0 {
        "❗ Poor gradient flow - action needed"
    } else {
        "❌ Critical gradient flow issues - immediate intervention required"
    }
}
