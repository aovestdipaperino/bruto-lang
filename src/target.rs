//! Target-platform shims for codegen.
//!
//! Pascal codegen embeds raw symbol names from the C runtime so the
//! generated IR can `extern` them through the linker. The names differ
//! between Apple's libc and glibc, so the codegen has to know which to
//! emit. We resolve that here, gated on `cfg!(target_os)` at codegen
//! build time — host = target for our compile-on-the-user's-machine
//! workflow.
//!
//! Windows is intentionally not handled yet: the MSVC CRT exposes
//! `stdin`/`stdout`/`stderr` as macros over `__acrt_iob_func(int)`
//! rather than as global symbols, so the codegen call sites would need
//! to emit a function call rather than a global load. That's a separate
//! piece of work tracked alongside the v1.0.0 Windows port.

/// Linker symbol for the C runtime's stdin FILE pointer.
#[cfg(target_os = "macos")]
pub const STDIN_SYM: &str = "__stdinp";
#[cfg(not(target_os = "macos"))]
pub const STDIN_SYM: &str = "stdin";

/// Linker symbol for the C runtime's stdout FILE pointer.
#[cfg(target_os = "macos")]
pub const STDOUT_SYM: &str = "__stdoutp";
#[cfg(not(target_os = "macos"))]
pub const STDOUT_SYM: &str = "stdout";

/// Linker symbol for the C runtime's stderr FILE pointer.
#[cfg(target_os = "macos")]
pub const STDERR_SYM: &str = "__stderrp";
#[cfg(not(target_os = "macos"))]
pub const STDERR_SYM: &str = "stderr";
