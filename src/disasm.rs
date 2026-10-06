/// Parses the textual assembly LLVM writes via `TargetMachine::write_to_file`
/// with `FileType::Assembly`, recovering the Pascal source line each
/// instruction belongs to from the embedded `.loc` directives DWARF-enabled
/// builds carry.
///
/// This is generic over any LLVM-backed [`crate::language::Language`] — the
/// `.loc` directive is LLVM's MC layer, not anything Pascal-specific — so it
/// lives here rather than in a particular language crate. The IDE's
/// Disassembly window (`bruto-ide`) is the only consumer today.
///
/// `.loc` syntax (GNU `as`-compatible, emitted by every LLVM backend):
/// `.loc <file-no> <line> [<column> [flags...]]`. It sets the "current"
/// source location for every line that follows until the next `.loc`, so
/// this is a simple one-pass scan carrying the last-seen line forward.
#[derive(Debug, Clone)]
pub struct AsmLine {
    /// The raw assembly text for this line (label, directive, or
    /// instruction) — `.loc` directives themselves are metadata, not
    /// shown here.
    pub text: String,
    /// 1-based Pascal source line this instruction was generated for, or
    /// `None` before the first `.loc` directive, when the build has no
    /// debug info (Retail), or when LLVM emitted `.loc <file> 0` for
    /// compiler-generated code with no source counterpart.
    pub source_line: Option<usize>,
}

/// Parse an LLVM-emitted `.s` listing into display lines tagged with their
/// Pascal source line. Blank lines are dropped (denser listing); every
/// other line — labels, directives, instructions — is kept and tagged
/// with the most recently seen `.loc` line.
pub fn parse(asm_text: &str) -> Vec<AsmLine> {
    let mut out = Vec::new();
    let mut current_line: Option<usize> = None;

    for raw in asm_text.lines() {
        let trimmed = raw.trim_start();
        if let Some(rest) = trimmed.strip_prefix(".loc") {
            // `.loc <file> <line> [<column> ...]` — second field is the
            // line; `0` means "no source" (compiler-generated code).
            let line_field = rest.split_whitespace().nth(1);
            current_line = line_field
                .and_then(|s| s.parse::<usize>().ok())
                .filter(|&n| n > 0);
            continue;
        }
        if trimmed.is_empty() {
            continue;
        }
        out.push(AsmLine {
            text: raw.to_string(),
            source_line: current_line,
        });
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_instructions_to_the_most_recent_loc_line() {
        let asm = "\
\t.text
\t.loc\t1 5 0
\tmovl\t$42, -4(%rbp)
\t.loc\t1 6 0
\tcallq\twriteln
";
        let lines = parse(asm);
        let texts: Vec<_> = lines.iter().map(|l| l.text.trim()).collect();
        assert_eq!(
            texts,
            vec![".text", "movl\t$42, -4(%rbp)", "callq\twriteln"]
        );
        assert_eq!(lines[0].source_line, None); // before any .loc
        assert_eq!(lines[1].source_line, Some(5));
        assert_eq!(lines[2].source_line, Some(6));
    }

    #[test]
    fn loc_line_zero_means_no_source() {
        let asm = "\t.loc\t1 0 0\n\tpushq\t%rbp\n";
        let lines = parse(asm);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].source_line, None);
    }

    #[test]
    fn blank_lines_are_dropped() {
        let asm = "\tmovl\t%eax, %ebx\n\n\n\tretq\n";
        let lines = parse(asm);
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn retail_build_with_no_loc_directives_has_no_mapping() {
        let asm = "\t.text\n\tmovl\t$1, %eax\n\tretq\n";
        let lines = parse(asm);
        assert!(lines.iter().all(|l| l.source_line.is_none()));
    }

    #[test]
    fn malformed_loc_is_ignored_without_panicking() {
        let asm = "\t.loc\tnotanumber\n\tnop\n";
        let lines = parse(asm);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].source_line, None);
    }
}
