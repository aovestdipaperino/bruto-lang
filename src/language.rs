//! The Language trait abstracts a programming language for the IDE.
//!
//! Implement this trait to add support for a new language. The IDE calls
//! these methods to get syntax highlighting, build executables, and
//! display language-specific information.

/// Result of a successful build.
pub struct BuildResult {
    /// Path to the compiled executable.
    pub exe_path: String,
    /// Path to the source file on disk (needed for DWARF debug info resolution).
    pub source_path: String,
    /// Path to the console output capture file (program writes here via compiled-in code).
    pub console_capture_path: String,
    /// `<exe>.bruto-prof`, written by the program at exit. `Some` only for
    /// profile builds.
    pub profile_path: Option<String>,
    /// `<exe>.bruto-prof-map`, written by codegen. `Some` only for profile builds.
    pub profile_map_path: Option<String>,
    /// Path to a `.s` text assembly listing of the compiled module, for
    /// the IDE's Disassembly window. `None` if the language's build
    /// doesn't produce one, or emission failed for this target.
    pub asm_path: Option<String>,
}

/// Whether a build carries debug info (for the debugger) or is an
/// optimized release binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BuildProfile {
    /// DWARF debug info, no optimization — what lldb needs.
    #[default]
    Debug,
    /// No debug info, optimized according to [`OptimizeFor`].
    Retail,
}

/// Optimization goal for a [`BuildProfile::Retail`] build.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptimizeFor {
    /// Smallest code (`-Os`).
    Size,
    /// Balanced size / speed (`-O2`).
    #[default]
    Both,
    /// Fastest code (`-O3`).
    Speed,
}

/// Compilation options chosen in the IDE's Build Options dialog (or
/// on the command line) and passed through to the language's build.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BuildOptions {
    pub profile: BuildProfile,
    pub optimize: OptimizeFor,
    /// Obfuscate the generated code (rename identifiers and obscure
    /// control flow). Only takes effect for Retail builds.
    pub obfuscate: bool,
}

impl BuildOptions {
    /// True when the build should emit DWARF debug info.
    pub fn debug_info(&self) -> bool {
        self.profile == BuildProfile::Debug
    }

    /// True when the obfuscation passes should run. Obfuscation applies to
    /// any profile when requested: the identifier rename is source-neutral,
    /// and the bogus-control-flow pass annotates its synthetic instructions
    /// with a line-0 debug location so a Debug build's disassembly stays
    /// aligned to the real Pascal lines.
    pub fn obfuscation_enabled(&self) -> bool {
        self.obfuscate
    }

    /// LLVM new-pass-manager pipeline to run, or `None` for an
    /// unoptimized (Debug) build. Debug builds ignore `optimize` so
    /// variables stay inspectable and stepping follows the source.
    pub fn pass_pipeline(&self) -> Option<&'static str> {
        match self.profile {
            BuildProfile::Debug => None,
            BuildProfile::Retail => Some(match self.optimize {
                OptimizeFor::Size => "default<Os>",
                OptimizeFor::Both => "default<O2>",
                OptimizeFor::Speed => "default<O3>",
            }),
        }
    }

    /// Same options with the profile forced to Debug — used when the
    /// debugger needs a build regardless of the user's selection.
    pub fn for_debugging(self) -> Self {
        Self {
            profile: BuildProfile::Debug,
            ..self
        }
    }

    /// Short human-readable description, e.g. `"Retail, optimized for speed"`.
    pub fn describe(&self) -> String {
        let base = match self.profile {
            BuildProfile::Debug => "Debug".to_string(),
            BuildProfile::Retail => format!("Retail, optimized for {}", self.optimize.as_str()),
        };
        if self.obfuscation_enabled() {
            format!("{base}, obfuscated")
        } else {
            base
        }
    }
}

impl BuildProfile {
    pub fn as_str(&self) -> &'static str {
        match self {
            BuildProfile::Debug => "debug",
            BuildProfile::Retail => "retail",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "debug" => Some(BuildProfile::Debug),
            "retail" | "release" => Some(BuildProfile::Retail),
            _ => None,
        }
    }
}

impl OptimizeFor {
    pub fn as_str(&self) -> &'static str {
        match self {
            OptimizeFor::Size => "size",
            OptimizeFor::Both => "both",
            OptimizeFor::Speed => "speed",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "size" => Some(OptimizeFor::Size),
            "both" => Some(OptimizeFor::Both),
            "speed" => Some(OptimizeFor::Speed),
            _ => None,
        }
    }
}

/// Status reported by [`BuildJob::poll`] each tick of the IDE's
/// progress dialog. `Pending` keeps the job alive; `Done`/`Failed`
/// terminate the dialog.
pub enum BuildPhase {
    /// Still working — the IDE redraws its progress dialog and polls
    /// again on the next tick. The string is shown to the user.
    Pending(String),
    Done(BuildResult),
    Failed(String),
}

/// A poll-driven build that the IDE drives cooperatively from its
/// modal progress dialog. Implementations chunk their work so heavy
/// phases (linking) can be polled with `try_wait` without blocking
/// the UI; cancelling = dropping the job, so the impl's `Drop` should
/// kill any spawned child processes.
pub trait BuildJob {
    /// Advance one step. The IDE polls this each redraw cycle until
    /// it returns `Done` or `Failed`.
    fn poll(&mut self) -> BuildPhase;
}

pub trait Language {
    /// Display name shown in the About dialog (e.g. "Mini-Pascal").
    fn name(&self) -> &str;

    /// File extension without the leading dot (e.g. "pas").
    fn file_extension(&self) -> &str;

    /// Sample program loaded into the editor on startup.
    fn sample_program(&self) -> &str;

    /// Create a syntax highlighter for the turbo-vision Editor.
    fn create_highlighter(&self) -> Box<dyn turbo_vision::views::syntax::SyntaxHighlighter>;

    /// Build the source as a poll-driven state machine. The IDE
    /// polls this from its progress dialog until [`BuildPhase::Done`]
    /// or [`BuildPhase::Failed`] is returned. Implementations should
    /// chunk work so the slow phases (linking) are pollable via
    /// `Child::try_wait` rather than blocking calls; the impl's
    /// `Drop` should kill any subprocesses to make Cancel real.
    fn build_job(&self, source: &str) -> Box<dyn BuildJob>;

    /// Like [`build_job`] but with an explicit on-disk path for the
    /// main source file. Languages that resolve imports (Pascal `uses`)
    /// from sibling files override this; otherwise the default just
    /// forwards to [`build_job`] and ignores the path.
    fn build_job_at(
        &self,
        source: &str,
        _source_path: Option<&std::path::Path>,
    ) -> Box<dyn BuildJob> {
        self.build_job(source)
    }

    /// Like [`build_job_at`] but with profiling instrumentation compiled
    /// in. The resulting [`BuildResult`] carries `profile_path` and
    /// `profile_map_path`. Default: a job that fails immediately, for
    /// languages without a profiler.
    fn profile_job_at(
        &self,
        source: &str,
        source_path: Option<&std::path::Path>,
    ) -> Box<dyn BuildJob> {
        let _ = (source, source_path);
        Box::new(UnsupportedJob)
    }

    /// Like [`build_job_at`] but honouring the user's [`BuildOptions`].
    /// Languages without optimization / debug-info control can keep the
    /// default, which ignores the options.
    fn build_job_with(
        &self,
        source: &str,
        source_path: Option<&std::path::Path>,
        _options: &BuildOptions,
    ) -> Box<dyn BuildJob> {
        self.build_job_at(source, source_path)
    }

    /// Convenience: drive `build_job` to completion synchronously.
    /// Used by callers that don't want progress info (CLI mode).
    fn build(&self, source: &str) -> Result<BuildResult, String> {
        let mut job = self.build_job(source);
        loop {
            match job.poll() {
                BuildPhase::Pending(_) => std::thread::sleep(std::time::Duration::from_millis(10)),
                BuildPhase::Done(r) => return Ok(r),
                BuildPhase::Failed(e) => return Err(e),
            }
        }
    }

    /// Read the profile written by a run of a `profile_job_at` build.
    fn load_profile(&self, result: &BuildResult) -> Result<crate::profile::Profile, String> {
        let (Some(p), Some(m)) = (&result.profile_path, &result.profile_map_path) else {
            return Err("this build was not a profile build".into());
        };
        crate::profile::Profile::load(p, m)
    }

    /// Return the set of 1-based line numbers where a breakpoint can validly
    /// be set (lines that produce executable code).  Used after a successful
    /// build to snap user-placed breakpoints to the nearest valid line.
    /// Default: every line is valid.
    fn valid_breakpoint_lines(&self, source: &str) -> std::collections::HashSet<usize> {
        let _ = source;
        (1..=source.lines().count()).collect()
    }
}

/// `BuildJob` for languages that do not implement a feature.
struct UnsupportedJob;

impl BuildJob for UnsupportedJob {
    fn poll(&mut self) -> BuildPhase {
        BuildPhase::Failed("profiling is not supported for this language".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_build_has_debug_info_and_no_passes() {
        let o = BuildOptions {
            profile: BuildProfile::Debug,
            optimize: OptimizeFor::Speed,
            obfuscate: false,
        };
        assert!(o.debug_info());
        assert_eq!(o.pass_pipeline(), None);
    }

    #[test]
    fn retail_build_maps_optimize_goal_to_pipeline() {
        let mk = |optimize| BuildOptions {
            profile: BuildProfile::Retail,
            optimize,
            obfuscate: false,
        };
        assert!(!mk(OptimizeFor::Both).debug_info());
        assert_eq!(mk(OptimizeFor::Size).pass_pipeline(), Some("default<Os>"));
        assert_eq!(mk(OptimizeFor::Both).pass_pipeline(), Some("default<O2>"));
        assert_eq!(mk(OptimizeFor::Speed).pass_pipeline(), Some("default<O3>"));
    }

    #[test]
    fn for_debugging_keeps_optimize_but_forces_debug() {
        let o = BuildOptions {
            profile: BuildProfile::Retail,
            optimize: OptimizeFor::Size,
            obfuscate: false,
        }
        .for_debugging();
        assert_eq!(o.profile, BuildProfile::Debug);
        assert_eq!(o.optimize, OptimizeFor::Size);
    }

    #[test]
    fn parse_round_trips() {
        for p in [BuildProfile::Debug, BuildProfile::Retail] {
            assert_eq!(BuildProfile::parse(p.as_str()), Some(p));
        }
        for o in [OptimizeFor::Size, OptimizeFor::Both, OptimizeFor::Speed] {
            assert_eq!(OptimizeFor::parse(o.as_str()), Some(o));
        }
        assert_eq!(BuildProfile::parse("Release"), Some(BuildProfile::Retail));
        assert_eq!(OptimizeFor::parse("fast"), None);
    }

    #[test]
    fn obfuscation_enabled_follows_the_flag_for_any_profile() {
        // The rename pass is source-neutral and BCF annotates its synthetic
        // instructions with a line-0 location, so obfuscation is valid on
        // Debug builds too — it tracks the flag, not the profile.
        for profile in [BuildProfile::Debug, BuildProfile::Retail] {
            let on = BuildOptions {
                profile,
                optimize: OptimizeFor::Both,
                obfuscate: true,
            };
            assert!(on.obfuscation_enabled());

            let off = BuildOptions {
                profile,
                optimize: OptimizeFor::Both,
                obfuscate: false,
            };
            assert!(!off.obfuscation_enabled());
        }
    }

    #[test]
    fn describe_mentions_obfuscation() {
        let o = BuildOptions {
            profile: BuildProfile::Retail,
            optimize: OptimizeFor::Speed,
            obfuscate: true,
        };
        assert!(o.describe().contains("obfuscated"));
        let plain = BuildOptions {
            profile: BuildProfile::Retail,
            optimize: OptimizeFor::Speed,
            obfuscate: false,
        };
        assert!(!plain.describe().contains("obfuscated"));
    }
}
