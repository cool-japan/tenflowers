#!/usr/bin/env python3
"""
Performance Regression Detection Tool for TenfloweRS Benchmarks

This tool analyzes criterion benchmark results and detects performance regressions
against configured thresholds. It supports:
- Criterion JSON output analysis
- Custom threshold configuration
- Historical trend detection
- Detailed regression reports
- GitHub Actions integration
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class RegressionSeverity(Enum):
    """Classification of regression severity"""
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class RegressionResult:
    """Result of a single benchmark regression check"""
    benchmark_name: str
    baseline_ns: float
    measured_ns: float
    regression_pct: float
    severity: RegressionSeverity
    threshold_pct: float


class PerformanceRegressionDetector:
    """Main detector class for analyzing performance regressions"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the detector with optional configuration file.

        Args:
            config_path: Path to performance-gates.json configuration
        """
        self.config = self._load_config(config_path)
        self.default_threshold = self.config.get(
            'global_settings', {}).get('default_regression_threshold_percent', 10.0)
        self.critical_threshold = self.config.get(
            'global_settings', {}).get('critical_operation_threshold_percent', 5.0)
        self.critical_ops = set(
            self.config.get('critical_operations', []))

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from JSON file"""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path) as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
        return {}

    def extract_criterion_results(self, criterion_dir: Path) -> List[Tuple[str, Dict]]:
        """
        Extract benchmark results from criterion JSON files.

        Args:
            criterion_dir: Path to criterion output directory

        Returns:
            List of (benchmark_name, benchmark_data) tuples
        """
        results = []

        for base_json in criterion_dir.glob('**/base/raw.json'):
            try:
                bench_name = base_json.parent.parent.name
                with open(base_json) as f:
                    data = json.load(f)
                results.append((bench_name, data))
            except Exception as e:
                print(f"Warning: Could not parse {base_json}: {e}")

        return results

    def compare_results(
        self,
        baseline_dir: Path,
        current_dir: Path
    ) -> List[RegressionResult]:
        """
        Compare baseline and current benchmark results.

        Args:
            baseline_dir: Path to baseline criterion output directory
            current_dir: Path to current criterion output directory

        Returns:
            List of RegressionResult objects
        """
        regressions = []

        for baseline_json in baseline_dir.glob('**/base/raw.json'):
            try:
                bench_name = baseline_json.parent.parent.name
                current_json = current_dir / \
                    baseline_json.relative_to(baseline_dir)

                if not current_json.exists():
                    continue

                with open(baseline_json) as f:
                    baseline_data = json.load(f)
                with open(current_json) as f:
                    current_data = json.load(f)

                baseline_mean = baseline_data.get(
                    'mean', {}).get('point_estimate', 0)
                current_mean = current_data.get(
                    'mean', {}).get('point_estimate', 0)

                if baseline_mean > 0:
                    regression_pct = (
                        (current_mean - baseline_mean) / baseline_mean) * 100.0

                    # Only report actual regressions (positive values)
                    if regression_pct > 0.5:  # Ignore tiny noise < 0.5%
                        threshold = self._get_threshold(bench_name)
                        severity = self._classify_severity(
                            regression_pct, threshold)
                        regressions.append(RegressionResult(
                            benchmark_name=bench_name,
                            baseline_ns=int(baseline_mean),
                            measured_ns=int(current_mean),
                            regression_pct=regression_pct,
                            severity=severity,
                            threshold_pct=threshold
                        ))

            except Exception as e:
                print(f"Warning: Could not compare benchmarks at {baseline_json}: {e}")

        return sorted(regressions, key=lambda x: -x.regression_pct)

    def _get_threshold(self, bench_name: str) -> float:
        """Get the regression threshold for a benchmark"""
        # Check if benchmark is critical
        is_critical = any(op in bench_name.lower()
                         for op in self.critical_ops)

        # Check for specific configuration
        benchmark_groups = self.config.get('benchmark_groups', {})
        for group_name, group_config in benchmark_groups.items():
            for op in group_config.get('operations', []):
                if op['name'] == bench_name:
                    return op.get('threshold_percent', self.default_threshold)

        # Return critical or default threshold
        return self.critical_threshold if is_critical else self.default_threshold

    def _classify_severity(self, regression_pct: float, threshold_pct: float) -> RegressionSeverity:
        """Classify the severity of a regression"""
        if regression_pct <= threshold_pct:
            return RegressionSeverity.WARNING
        else:
            return RegressionSeverity.CRITICAL

    def generate_markdown_report(self, regressions: List[RegressionResult]) -> str:
        """Generate a markdown report of regressions"""
        if not regressions:
            return "✅ **No performance regressions detected!**"

        lines = ["## 📊 Performance Regression Analysis\n"]

        critical_regressions = [
            r for r in regressions if r.severity == RegressionSeverity.CRITICAL]
        warning_regressions = [
            r for r in regressions if r.severity == RegressionSeverity.WARNING]

        if critical_regressions:
            lines.append("### ❌ Critical Regressions (exceeding threshold)\n")
            for reg in critical_regressions:
                lines.append(
                    f"- **{reg.benchmark_name}**: {reg.regression_pct:+.2f}% "
                    f"(baseline: {reg.baseline_ns}ns → current: {reg.measured_ns}ns, "
                    f"threshold: {reg.threshold_pct:.1f}%)"
                )

        if warning_regressions:
            lines.append("\n### ⚠️ Warnings (within acceptable threshold)\n")
            for reg in warning_regressions:
                lines.append(
                    f"- **{reg.benchmark_name}**: {reg.regression_pct:+.2f}% "
                    f"(baseline: {reg.baseline_ns}ns → current: {reg.measured_ns}ns, "
                    f"threshold: {reg.threshold_pct:.1f}%)"
                )

        # Summary statistics
        lines.append("\n### Summary\n")
        lines.append(
            f"- Total regressions: {len(regressions)}")
        lines.append(f"- Critical: {len(critical_regressions)}")
        lines.append(f"- Warnings: {len(warning_regressions)}")

        if critical_regressions:
            lines.append(f"\n❌ **Build failed: {len(critical_regressions)} critical regression(s) detected**")
        else:
            lines.append(f"\n✅ All regressions within acceptable thresholds")

        return '\n'.join(lines)

    def generate_json_report(self, regressions: List[RegressionResult]) -> Dict:
        """Generate a JSON report of regressions"""
        return {
            "total_regressions": len(regressions),
            "critical_count": sum(1 for r in regressions if r.severity == RegressionSeverity.CRITICAL),
            "warning_count": sum(1 for r in regressions if r.severity == RegressionSeverity.WARNING),
            "regressions": [
                {
                    "benchmark": r.benchmark_name,
                    "regression_pct": round(r.regression_pct, 2),
                    "baseline_ns": r.baseline_ns,
                    "measured_ns": r.measured_ns,
                    "threshold_pct": r.threshold_pct,
                    "severity": r.severity.value
                }
                for r in regressions
            ]
        }


def main():
    parser = argparse.ArgumentParser(
        description="Detect performance regressions in benchmarks"
    )
    parser.add_argument(
        '--baseline-dir',
        type=Path,
        default=Path('target/criterion'),
        help='Path to baseline criterion directory'
    )
    parser.add_argument(
        '--current-dir',
        type=Path,
        default=Path('target/criterion'),
        help='Path to current criterion directory'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('.github/performance-gates.json'),
        help='Path to performance gates configuration'
    )
    parser.add_argument(
        '--output-format',
        choices=['markdown', 'json', 'both'],
        default='markdown',
        help='Output format for report'
    )
    parser.add_argument(
        '--github-output',
        action='store_true',
        help='Write to GITHUB_OUTPUT for CI integration'
    )
    parser.add_argument(
        '--json-report',
        type=Path,
        help='Write JSON report to file'
    )

    args = parser.parse_args()

    # Initialize detector
    detector = PerformanceRegressionDetector(
        config_path=str(args.config) if args.config.exists() else None
    )

    # Analyze regressions
    regressions = detector.compare_results(args.baseline_dir, args.current_dir)

    # Generate reports
    markdown_report = detector.generate_markdown_report(regressions)
    json_report = detector.generate_json_report(regressions)

    # Output results
    if args.output_format in ['markdown', 'both']:
        print(markdown_report)

    if args.output_format in ['json', 'both']:
        print(json.dumps(json_report, indent=2))

    # Write JSON report to file
    if args.json_report:
        with open(args.json_report, 'w') as f:
            json.dump(json_report, f, indent=2)

    # Write to GitHub Output
    if args.github_output:
        github_output = os.environ.get('GITHUB_OUTPUT', '/dev/null')
        with open(github_output, 'a') as f:
            critical_count = json_report.get('critical_count', 0)
            warning_count = json_report.get('warning_count', 0)
            f.write(f"regression_detected={'true' if critical_count > 0 else 'false'}\n")
            f.write(f"critical_count={critical_count}\n")
            f.write(f"warning_count={warning_count}\n")
            f.write(f"regression_summary="
                   f"Found {json_report['total_regressions']} regressions, "
                   f"{critical_count} critical, "
                   f"{warning_count} warnings\n")

    # Exit with error if critical regressions found
    if json_report.get('critical_count', 0) > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
