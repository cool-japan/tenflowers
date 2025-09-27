//! Alert management system
//!
//! This module handles alert rules, notifications, and alert lifecycle management
//! for the ultra performance monitoring system.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};

/// Alert management system
#[allow(dead_code)]
pub struct AlertManager {
    /// Active alerts
    pub(crate) active_alerts: Vec<PerformanceAlert>,
    /// Alert rules and thresholds
    pub(crate) alert_rules: Vec<AlertRule>,
    /// Alert history
    pub(crate) alert_history: VecDeque<AlertEvent>,
    /// Notification channels
    pub(crate) notification_channels: Vec<NotificationChannel>,
    /// Alert suppression rules
    pub(crate) suppression_rules: Vec<SuppressionRule>,
}

/// Performance alert definition
#[derive(Debug, Clone)]
pub struct PerformanceAlert {
    /// Alert ID
    pub alert_id: String,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert message
    pub message: String,
    /// Alert trigger time
    pub trigger_time: SystemTime,
    /// Alert rule that triggered
    pub triggered_by: String,
    /// Related metrics
    pub related_metrics: Vec<String>,
    /// Suggested actions
    pub suggested_actions: Vec<String>,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Alert rule configuration
#[derive(Debug, Clone)]
pub struct AlertRule {
    /// Rule ID
    pub rule_id: String,
    /// Rule name
    pub rule_name: String,
    /// Metric to monitor
    pub metric_name: String,
    /// Threshold value
    pub threshold: f64,
    /// Comparison operator
    pub operator: ComparisonOperator,
    /// Evaluation window
    pub evaluation_window: Duration,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Rule enabled
    pub enabled: bool,
}

/// Comparison operators for alert rules
#[derive(Debug, Clone, Copy)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equal,
    NotEqual,
    GreaterThanOrEqual,
    LessThanOrEqual,
}

/// Alert event record
#[derive(Debug, Clone)]
pub struct AlertEvent {
    /// Event ID
    pub event_id: String,
    /// Event type
    pub event_type: AlertEventType,
    /// Alert involved
    pub alert: PerformanceAlert,
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Event details
    pub details: HashMap<String, String>,
}

/// Alert event types
#[derive(Debug, Clone, Copy)]
pub enum AlertEventType {
    AlertTriggered,
    AlertResolved,
    AlertSuppressed,
    AlertEscalated,
}

/// Notification channel configuration
#[derive(Debug, Clone)]
pub struct NotificationChannel {
    /// Channel ID
    pub channel_id: String,
    /// Channel type
    pub channel_type: NotificationChannelType,
    /// Channel configuration
    pub config: HashMap<String, String>,
    /// Enabled status
    pub enabled: bool,
}

/// Notification channel types
#[derive(Debug, Clone)]
pub enum NotificationChannelType {
    Email { recipients: Vec<String> },
    Slack { webhook_url: String },
    PagerDuty { integration_key: String },
    Webhook { url: String },
    Console,
}

/// Alert suppression rule
#[derive(Debug, Clone)]
pub struct SuppressionRule {
    /// Rule ID
    pub rule_id: String,
    /// Metric patterns to suppress
    pub metric_patterns: Vec<String>,
    /// Suppression duration
    pub duration: Duration,
    /// Active status
    pub active: bool,
}

impl AlertManager {
    /// Create new alert manager
    pub(crate) fn new() -> Self {
        Self {
            active_alerts: Vec::new(),
            alert_rules: Vec::new(),
            alert_history: VecDeque::new(),
            notification_channels: Vec::new(),
            suppression_rules: Vec::new(),
        }
    }

    /// Add alert rule
    pub(crate) fn add_alert_rule(&mut self, rule: AlertRule) {
        self.alert_rules.push(rule);
    }

    /// Remove alert rule
    #[allow(dead_code)]
    pub(crate) fn remove_alert_rule(&mut self, rule_id: &str) {
        self.alert_rules.retain(|rule| rule.rule_id != rule_id);
    }

    /// Trigger alert
    #[allow(dead_code)]
    pub(crate) fn trigger_alert(&mut self, alert: PerformanceAlert) {
        let event = AlertEvent {
            event_id: format!("evt_{}", alert.alert_id),
            event_type: AlertEventType::AlertTriggered,
            alert: alert.clone(),
            timestamp: SystemTime::now(),
            details: HashMap::new(),
        };

        self.active_alerts.push(alert);
        self.alert_history.push_back(event);

        // Limit history size
        while self.alert_history.len() > 10000 {
            self.alert_history.pop_front();
        }
    }

    /// Resolve alert
    #[allow(dead_code)]
    pub(crate) fn resolve_alert(&mut self, alert_id: &str) {
        if let Some(pos) = self
            .active_alerts
            .iter()
            .position(|a| a.alert_id == alert_id)
        {
            let alert = self.active_alerts.remove(pos);

            let event = AlertEvent {
                event_id: format!("evt_resolved_{}", alert_id),
                event_type: AlertEventType::AlertResolved,
                alert,
                timestamp: SystemTime::now(),
                details: HashMap::new(),
            };

            self.alert_history.push_back(event);
        }
    }

    /// Add notification channel
    #[allow(dead_code)]
    pub(crate) fn add_notification_channel(&mut self, channel: NotificationChannel) {
        self.notification_channels.push(channel);
    }

    /// Add suppression rule
    #[allow(dead_code)]
    pub(crate) fn add_suppression_rule(&mut self, rule: SuppressionRule) {
        self.suppression_rules.push(rule);
    }

    /// Get active alerts
    pub(crate) fn get_active_alerts(&self) -> &[PerformanceAlert] {
        &self.active_alerts
    }

    /// Get alert history
    #[allow(dead_code)]
    pub(crate) fn get_alert_history(&self, count: usize) -> Vec<&AlertEvent> {
        self.alert_history.iter().rev().take(count).collect()
    }
}

impl Default for AlertRule {
    fn default() -> Self {
        Self {
            rule_id: String::new(),
            rule_name: String::new(),
            metric_name: String::new(),
            threshold: 0.0,
            operator: ComparisonOperator::GreaterThan,
            evaluation_window: Duration::from_secs(60),
            severity: AlertSeverity::Warning,
            enabled: true,
        }
    }
}
