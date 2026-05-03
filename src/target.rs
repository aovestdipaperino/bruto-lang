//! Target-platform shims for codegen.
//!
//! Pascal codegen needs to fetch `FILE *` pointers for stdin/stdout/stderr
//! from the host C runtime. The mechanism is platform-dependent:
//!
//! - **Apple libc / glibc**: stdio streams are *global symbols*. The
//!   codegen emits an `extern global FILE *<sym>` and `load`s it.
//! - **MSVC CRT (Windows)**: stdio streams are *macros* expanding to
//!   `__acrt_iob_func(int)`. The codegen emits a function declaration
//!   and `call`s it with 0/1/2 to select stdin/stdout/stderr.
//!
//! Use [`emit_load_stdio`] from the codegen rather than dealing with
//! either form directly — it picks the right shape based on
//! `cfg!(target_env)` at codegen-binary build time.

use inkwell::AddressSpace;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::{Linkage, Module};
use inkwell::values::PointerValue;

/// One of the three standard streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stdio {
    Stdin,
    Stdout,
    Stderr,
}

impl Stdio {
    /// Index for `__acrt_iob_func(int)` on MSVC. Only consumed inside the
    /// MSVC arm of [`emit_load_stdio`]; dead on every other target.
    #[allow(dead_code)]
    fn msvc_index(self) -> u64 {
        match self {
            Stdio::Stdin => 0,
            Stdio::Stdout => 1,
            Stdio::Stderr => 2,
        }
    }

    /// Linker symbol on Apple libc / glibc.
    fn unix_symbol(self) -> &'static str {
        match self {
            Stdio::Stdin => STDIN_SYM,
            Stdio::Stdout => STDOUT_SYM,
            Stdio::Stderr => STDERR_SYM,
        }
    }
}

/// Linker symbol for the C runtime's stdin FILE pointer (Unix-y targets).
#[cfg(target_os = "macos")]
pub const STDIN_SYM: &str = "__stdinp";
#[cfg(all(not(target_os = "macos"), not(target_env = "msvc")))]
pub const STDIN_SYM: &str = "stdin";
#[cfg(target_env = "msvc")]
pub const STDIN_SYM: &str = ""; // unused — MSVC goes through the function path

/// Linker symbol for the C runtime's stdout FILE pointer.
#[cfg(target_os = "macos")]
pub const STDOUT_SYM: &str = "__stdoutp";
#[cfg(all(not(target_os = "macos"), not(target_env = "msvc")))]
pub const STDOUT_SYM: &str = "stdout";
#[cfg(target_env = "msvc")]
pub const STDOUT_SYM: &str = "";

/// Linker symbol for the C runtime's stderr FILE pointer.
#[cfg(target_os = "macos")]
pub const STDERR_SYM: &str = "__stderrp";
#[cfg(all(not(target_os = "macos"), not(target_env = "msvc")))]
pub const STDERR_SYM: &str = "stderr";
#[cfg(target_env = "msvc")]
pub const STDERR_SYM: &str = "";

/// Emit IR that loads the requested stdio FILE pointer at runtime.
///
/// On Unix-y targets this declares an `extern` global and loads it; on
/// MSVC it declares `FILE *__acrt_iob_func(int)` and calls it. The
/// returned pointer has type `ptr` (LLVM's opaque pointer) which the
/// caller can use as the first argument to `fprintf`, `fgets`, etc.
pub fn emit_load_stdio<'ctx>(
    builder: &Builder<'ctx>,
    module: &Module<'ctx>,
    context: &'ctx Context,
    which: Stdio,
) -> PointerValue<'ctx> {
    let ptr_ty = context.ptr_type(AddressSpace::default());

    #[cfg(target_env = "msvc")]
    {
        let i32_ty = context.i32_type();
        let acrt = module.get_function("__acrt_iob_func").unwrap_or_else(|| {
            let ft = ptr_ty.fn_type(&[i32_ty.into()], false);
            module.add_function("__acrt_iob_func", ft, None)
        });
        builder
            .build_call(
                acrt,
                &[i32_ty.const_int(which.msvc_index(), false).into()],
                "stdio_fp",
            )
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_pointer_value()
    }

    #[cfg(not(target_env = "msvc"))]
    {
        let sym = which.unix_symbol();
        let g = module.get_global(sym).unwrap_or_else(|| {
            let g = module.add_global(ptr_ty, None, sym);
            g.set_externally_initialized(true);
            g.set_linkage(Linkage::External);
            g
        });
        builder
            .build_load(ptr_ty, g.as_pointer_value(), "stdio_fp")
            .unwrap()
            .into_pointer_value()
    }
}
