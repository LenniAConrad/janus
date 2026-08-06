//! Fallible scoped-thread admission with exact synchronous fallback.
//!
//! Inference and maintenance code often partitions disjoint output slices
//! across scoped helpers. The standard `Scope::spawn` panics when the operating
//! system refuses a thread. This module retains each task on the caller until a
//! named `Builder::spawn_scoped` succeeds, then transfers it through a zero-slot
//! rendezvous channel. A refusal executes the same task synchronously, keeping
//! output ownership, scalar order, and completion semantics unchanged.

use std::sync::mpsc;
use std::thread::{Builder, Scope};

/// Used for starting one scoped task or executing it on the caller after a
/// thread-admission failure.
///
/// The worker starts by waiting on a zero-capacity channel. Consequently the
/// task, including any borrowed mutable output slice, remains owned by this
/// function until `spawn_scoped` has returned success. If the new worker
/// disappears before the rendezvous, `SendError` returns the task and the
/// caller executes it instead.
///
/// # Arguments
///
/// * `scope` - lexical thread scope owning any successfully started helper
/// * `name` - diagnostic thread name visible to host tooling
/// * `task` - complete disjoint unit of work
pub(crate) fn spawn_scoped_or_run<'scope, 'env, F>(
    scope: &'scope Scope<'scope, 'env>,
    name: &str,
    task: F,
) where
    F: FnOnce() + Send + 'scope,
{
    let (sender, receiver) = mpsc::sync_channel::<F>(0);
    let spawned = Builder::new()
        .name(name.to_owned())
        .spawn_scoped(scope, move || {
            if let Ok(task) = receiver.recv() {
                task();
            }
        });
    match spawned {
        Ok(_) => {
            if let Err(error) = sender.send(task) {
                (error.0)();
            }
        }
        Err(_) => task(),
    }
}

