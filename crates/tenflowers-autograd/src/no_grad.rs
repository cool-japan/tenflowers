//! No-gradient context for disabling gradient computation
//!
//! This module provides utilities to temporarily disable gradient tracking,
//! which is useful for inference or when you want to reduce memory usage
//! and improve performance during evaluation.

use std::cell::RefCell;

// Thread-local stack to track nested no_grad contexts and global state
thread_local! {
    static GRAD_STACK: RefCell<Vec<bool>> = const { RefCell::new(Vec::new()) };
    static THREAD_GRAD_ENABLED: RefCell<bool> = const { RefCell::new(true) };
}

/// Check if gradient computation is currently enabled
pub fn is_grad_enabled() -> bool {
    GRAD_STACK.with(|stack| {
        let stack = stack.borrow();
        if let Some(&last) = stack.last() {
            last
        } else {
            THREAD_GRAD_ENABLED.with(|enabled| *enabled.borrow())
        }
    })
}

/// Set the global gradient computation state
pub fn set_grad_enabled(enabled: bool) {
    THREAD_GRAD_ENABLED.with(|thread_enabled| {
        *thread_enabled.borrow_mut() = enabled;
    });
}

/// RAII guard for no-gradient context
pub struct NoGradGuard {
    previous_state: bool,
}

impl Default for NoGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl NoGradGuard {
    pub fn new() -> Self {
        let previous_state = is_grad_enabled();
        GRAD_STACK.with(|stack| {
            stack.borrow_mut().push(false);
        });

        NoGradGuard { previous_state }
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        GRAD_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            stack.pop();
            // If the stack is now empty, restore the previous state
            if stack.is_empty() {
                drop(stack); // Release the borrow before calling set_grad_enabled
                set_grad_enabled(self.previous_state);
            }
        });
    }
}

/// RAII guard for enable-gradient context
pub struct EnableGradGuard {
    previous_state: bool,
}

impl Default for EnableGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl EnableGradGuard {
    pub fn new() -> Self {
        let previous_state = is_grad_enabled();
        GRAD_STACK.with(|stack| {
            stack.borrow_mut().push(true);
        });

        EnableGradGuard { previous_state }
    }
}

impl Drop for EnableGradGuard {
    fn drop(&mut self) {
        GRAD_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            stack.pop();
            // If the stack is now empty, restore the previous state
            if stack.is_empty() {
                drop(stack); // Release the borrow before calling set_grad_enabled
                set_grad_enabled(self.previous_state);
            }
        });
    }
}

/// Execute a closure with gradient computation disabled
pub fn no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = NoGradGuard::new();
    f()
}

/// Execute a closure with gradient computation enabled
pub fn enable_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = EnableGradGuard::new();
    f()
}

/// Macro for no-grad context (convenience)
#[macro_export]
macro_rules! no_grad {
    ($($tt:tt)*) => {
        $crate::no_grad::no_grad(|| {
            $($tt)*
        })
    };
}

/// Macro for enable-grad context (convenience)
#[macro_export]
macro_rules! enable_grad {
    ($($tt:tt)*) => {
        $crate::no_grad::enable_grad(|| {
            $($tt)*
        })
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test setup function to ensure clean state
    fn setup() {
        // Reset thread-local state and clear thread-local stack
        set_grad_enabled(true);
        GRAD_STACK.with(|stack| {
            stack.borrow_mut().clear();
        });
    }

    #[test]
    fn test_global_grad_state() {
        setup();
        // Initially enabled
        assert!(is_grad_enabled());

        // Disable globally
        set_grad_enabled(false);
        assert!(!is_grad_enabled());

        // Re-enable
        set_grad_enabled(true);
        assert!(is_grad_enabled());
    }

    #[test]
    fn test_no_grad_context() {
        setup();
        assert!(is_grad_enabled());

        let result = no_grad(|| {
            assert!(!is_grad_enabled());
            42
        });

        assert_eq!(result, 42);
        assert!(is_grad_enabled());
    }

    #[test]
    fn test_enable_grad_context() {
        setup();
        set_grad_enabled(false);
        assert!(!is_grad_enabled());

        let result = enable_grad(|| {
            assert!(is_grad_enabled());
            42
        });

        assert_eq!(result, 42);
        assert!(!is_grad_enabled());

        // Reset for other tests
        set_grad_enabled(true);
    }

    #[test]
    fn test_nested_contexts() {
        setup();
        assert!(is_grad_enabled());

        no_grad(|| {
            assert!(!is_grad_enabled());

            enable_grad(|| {
                assert!(is_grad_enabled());

                no_grad(|| {
                    assert!(!is_grad_enabled());
                });

                assert!(is_grad_enabled());
            });

            assert!(!is_grad_enabled());
        });

        assert!(is_grad_enabled());
    }

    #[test]
    fn test_no_grad_guard() {
        setup();
        assert!(is_grad_enabled());

        {
            let _guard = NoGradGuard::new();
            assert!(!is_grad_enabled());
        }

        assert!(is_grad_enabled());
    }

    #[test]
    fn test_enable_grad_guard() {
        setup();
        set_grad_enabled(false);
        assert!(!is_grad_enabled());

        {
            let _guard = EnableGradGuard::new();
            assert!(is_grad_enabled());
        }

        assert!(!is_grad_enabled());

        // Reset
        set_grad_enabled(true);
    }
}
