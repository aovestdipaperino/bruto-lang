/// The Language trait abstracts a programming language for the IDE.
///
/// Implement this trait to add support for a new language. The IDE calls
/// these methods to get syntax highlighting, build executables, and
/// display language-specific information.

/// Result of a successful build.
pub struct BuildResult {
    /// Path to the compiled executable.
    pub exe_path: String,
    /// Path to the source file on disk (needed for DWARF debug info resolution).
    pub source_path: String,
    /// Path to the console output capture file (program writes here via compiled-in code).
    pub console_capture_path: String,
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
    fn build_job_at(&self, source: &str, _source_path: Option<&std::path::Path>) -> Box<dyn BuildJob> {
        self.build_job(source)
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

    /// Return the set of 1-based line numbers where a breakpoint can validly
    /// be set (lines that produce executable code).  Used after a successful
    /// build to snap user-placed breakpoints to the nearest valid line.
    /// Default: every line is valid.
    fn valid_breakpoint_lines(&self, source: &str) -> std::collections::HashSet<usize> {
        let _ = source;
        (1..=source.lines().count()).collect()
    }
}
