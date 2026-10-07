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
///
/// Besides the mapping, the parser classifies every line (function entry,
/// branch target, instruction, data) and splits it into mnemonic /
/// operands / trailing comment so the window can lay it out in columns.
/// Assembler bookkeeping that means nothing to a reader — CFI directives,
/// alignment, symbol visibility, the DWARF sections themselves, and the
/// `Ltmp` / `Lfunc_begin` labels that only exist to anchor debug info —
/// is dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsmKind {
    /// A section switch to a non-debug section (`.section __TEXT,__cstring`);
    /// `operands` holds the section name.
    Section,
    /// A global code label — the entry of a function / procedure.
    Function,
    /// A local branch target inside a function (`LBB0_1`, `.LBB0_1`).
    Block,
    /// A label on data (string literals, constant pools, globals).
    Data,
    /// A machine instruction.
    Instruction,
    /// A data-emitting directive kept for display (`.asciz`, `.quad`, ...).
    Directive,
}

#[derive(Debug, Clone)]
pub struct AsmLine {
    pub kind: AsmKind,
    /// Label name for label kinds, mnemonic for instructions, the
    /// directive (with its dot) for directives, empty for sections.
    pub mnemonic: String,
    /// Instruction operands / directive arguments / section name, with
    /// tabs normalised to spaces. Empty for labels.
    pub operands: String,
    /// The assembler comment LLVM appended (`=0x2a`, `Folded Spill`, loop
    /// annotations on block labels), without its comment marker.
    pub comment: Option<String>,
    /// 1-based Pascal source line this line was generated for, or `None`
    /// before the first `.loc` directive, when the build has no debug info
    /// (Retail), or when LLVM emitted `.loc <file> 0` for
    /// compiler-generated code with no source counterpart. Labels take the
    /// line of the first instruction they introduce, since `.loc` follows
    /// the label in LLVM's output.
    pub source_line: Option<usize>,
}

/// Directives that are pure assembler bookkeeping and are never shown.
const HIDDEN_DIRECTIVES: &[&str] = &[
    ".p2align",
    ".align",
    ".balign",
    ".globl",
    ".global",
    ".private_extern",
    ".weak_definition",
    ".weak_reference",
    ".weak",
    ".hidden",
    ".type",
    ".size",
    ".file",
    ".build_version",
    ".macosx_version_min",
    ".ident",
    ".addrsig",
    ".addrsig_sym",
    ".subsections_via_symbols",
    ".loh",
    ".data_region",
    ".end_data_region",
    ".no_dead_strip",
    ".alt_entry",
];

/// Prefixes of local labels LLVM emits only to anchor debug info, CFI, or
/// linker hints — never a branch target a reader cares about.
const HIDDEN_LABEL_PREFIXES: &[&str] = &[
    "Ltmp",
    "Lfunc_",
    "Lloh",
    "Lcfi",
    "Lsection",
    "Ldebug",
    "Linfo",
    "Lcu_",
    "Lline",
    "Lnames",
    "Lset",
    "Lexception",
    "Lstring",
    "Laddr",
    "Lrnglists",
    "Lloclists",
    "Lstr_offsets",
];

/// Parse an LLVM-emitted `.s` listing into classified display lines
/// tagged with their Pascal source line. Blank lines, comment-only lines,
/// bookkeeping directives and everything inside DWARF sections are
/// dropped.
pub fn parse(asm_text: &str) -> Vec<AsmLine> {
    let mut out: Vec<AsmLine> = Vec::new();
    let mut current_line: Option<usize> = None;
    let mut in_debug = false;
    let mut in_text = true;

    for raw in asm_text.lines() {
        let (code, comment) = split_comment(raw);
        let code = code.trim();
        if let Some(rest) = code.strip_prefix(".loc")
            && rest.starts_with(char::is_whitespace)
        {
            // `.loc <file> <line> [<column> ...]` — second field is the
            // line; `0` means "no source" (compiler-generated).
            current_line = rest
                .split_whitespace()
                .nth(1)
                .and_then(|s| s.parse::<usize>().ok())
                .filter(|&n| n > 0);
            continue;
        }
        if code.is_empty() {
            continue;
        }

        if let Some(section) = section_name(code) {
            in_debug = is_debug_section(&section);
            in_text = is_text_section(&section);
            if !in_debug {
                out.push(AsmLine {
                    kind: AsmKind::Section,
                    mnemonic: String::new(),
                    operands: section,
                    comment: None,
                    source_line: None,
                });
            }
            continue;
        }
        if in_debug {
            continue;
        }

        let line = if let Some(label) = code.strip_suffix(':').filter(|l| is_label(l)) {
            let Some(kind) = label_kind(label, in_text) else {
                continue;
            };
            AsmLine {
                kind,
                mnemonic: label.to_string(),
                operands: String::new(),
                // Mach-O labels carry `; @fact` / `; @.str` — the IR name,
                // which repeats the label and adds nothing.
                comment: comment.filter(|c| !c.starts_with('@')),
                source_line: current_line,
            }
        } else {
            let (mnemonic, operands) = split_mnemonic(code);
            let kind = if mnemonic.starts_with('.') {
                if mnemonic.starts_with(".cfi_") || HIDDEN_DIRECTIVES.contains(&mnemonic) {
                    continue;
                }
                AsmKind::Directive
            } else {
                AsmKind::Instruction
            };
            AsmLine {
                kind,
                mnemonic: mnemonic.to_string(),
                operands,
                comment,
                source_line: current_line,
            }
        };
        out.push(line);
    }

    // A label precedes the `.loc` of the code it introduces, so the scan
    // above tagged it with the *previous* statement's line. Re-tag each
    // label with the first instruction that follows it.
    let mut next_instr_line = None;
    for line in out.iter_mut().rev() {
        match line.kind {
            AsmKind::Instruction => next_instr_line = line.source_line,
            AsmKind::Function | AsmKind::Block => line.source_line = next_instr_line,
            AsmKind::Section | AsmKind::Data | AsmKind::Directive => next_instr_line = None,
        }
    }

    out
}

/// Split a raw line into code and trailing comment. LLVM's comment markers
/// differ per target: `;` (AArch64 Mach-O), `##` / `#` (x86), `//`
/// (AArch64 ELF). A `#` only starts a comment when followed by a space or
/// another `#`, so AArch64 immediates like `#16` stay in the code. Quoted
/// strings (`.asciz "a;b"`) are skipped.
fn split_comment(raw: &str) -> (&str, Option<String>) {
    let bytes = raw.as_bytes();
    let mut in_quote = false;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if in_quote {
            if b == b'\\' {
                i += 1;
            } else if b == b'"' {
                in_quote = false;
            }
        } else if b == b'"' {
            in_quote = true;
        } else {
            let next = bytes.get(i + 1).copied();
            let marker_len = match b {
                b';' => Some(1),
                b'/' if next == Some(b'/') => Some(2),
                b'#' if next == Some(b'#') => Some(2),
                b'#' if next.is_none_or(|n| n == b' ' || n == b'\t') => Some(1),
                _ => None,
            };
            if let Some(len) = marker_len {
                let text = raw[i + len..].trim();
                let comment = (!text.is_empty()).then(|| text.to_string());
                return (&raw[..i], comment);
            }
        }
        i += 1;
    }
    (raw, None)
}

/// Section name if `code` switches sections: `.section <name>[,flags]`,
/// or one of the shorthand directives.
fn section_name(code: &str) -> Option<String> {
    let (mnemonic, operands) = split_mnemonic(code);
    match mnemonic {
        ".section" => {
            // Mach-O: `__SEG,__sect,type,attrs`; ELF: `.name,"flags",@type`.
            // Keep segment+section for Mach-O, the name alone for ELF.
            let mut parts = operands.split(',').map(str::trim);
            let first = parts.next().unwrap_or_default();
            Some(if first.starts_with("__") {
                match parts.next() {
                    Some(sect) => format!("{first},{sect}"),
                    None => first.to_string(),
                }
            } else {
                first.to_string()
            })
        }
        ".text" | ".data" | ".bss" | ".cstring" | ".const" | ".rodata" | ".literal4"
        | ".literal8" | ".literal16" => Some(mnemonic.to_string()),
        _ => None,
    }
}

fn is_debug_section(name: &str) -> bool {
    name.contains("DWARF") || name.contains("debug") || name.starts_with(".note")
}

fn is_text_section(name: &str) -> bool {
    name == ".text" || name.starts_with(".text.") || name.ends_with(",__text")
}

/// True when `s` (the text before a trailing `:`) is a plain label rather
/// than, say, an instruction operand that happens to end in a colon.
fn is_label(s: &str) -> bool {
    !s.is_empty() && !s.contains(char::is_whitespace)
        || (s.starts_with('"') && s.ends_with('"') && s.len() >= 2)
}

/// Classify a label, or `None` for debug-only labels that are hidden.
fn label_kind(label: &str, in_text: bool) -> Option<AsmKind> {
    let local = label
        .strip_prefix(".L")
        .map(|rest| format!("L{rest}"))
        .or_else(|| (label.starts_with('L') || label.starts_with('l')).then(|| label.to_string()));
    match local {
        Some(l) if HIDDEN_LABEL_PREFIXES.iter().any(|p| l.starts_with(p)) => None,
        Some(l) if l.starts_with("LBB") => Some(AsmKind::Block),
        Some(_) => Some(AsmKind::Data),
        None if in_text => Some(AsmKind::Function),
        None => Some(AsmKind::Data),
    }
}

/// Split `code` into its first whitespace-delimited token and the rest,
/// with internal tabs collapsed to single spaces.
fn split_mnemonic(code: &str) -> (&str, String) {
    match code.find(char::is_whitespace) {
        Some(pos) => {
            let rest = code[pos..].trim();
            (&code[..pos], rest.replace('\t', " "))
        }
        None => (code, String::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(lines: &[AsmLine]) -> Vec<AsmKind> {
        lines.iter().map(|l| l.kind).collect()
    }

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
        assert_eq!(
            kinds(&lines),
            vec![AsmKind::Section, AsmKind::Instruction, AsmKind::Instruction]
        );
        assert_eq!(lines[1].mnemonic, "movl");
        assert_eq!(lines[1].operands, "$42, -4(%rbp)");
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
    fn blank_and_comment_only_lines_are_dropped() {
        let asm = "\tmovl\t%eax, %ebx\n\n; %bb.0:\n                ; -- End function\n\tretq\n";
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

    #[test]
    fn classifies_labels_and_hides_debug_anchors() {
        let asm = "\
\t.section\t__TEXT,__text,regular,pure_instructions
\t.globl\t_fact                           ; -- Begin function fact
\t.p2align\t2
_fact:                                  ; @fact
Lfunc_begin0:
\t.loc\t1 2 0
\t.cfi_startproc
\tsub\tsp, sp, #16
Ltmp1:
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
\t.loc\t1 3 0
\tldr\tw8, [sp, #12]
";
        let lines = parse(asm);
        assert_eq!(
            kinds(&lines),
            vec![
                AsmKind::Section,
                AsmKind::Function,
                AsmKind::Instruction,
                AsmKind::Block,
                AsmKind::Instruction,
            ]
        );
        assert_eq!(lines[0].operands, "__TEXT,__text");
        assert_eq!(lines[1].mnemonic, "_fact");
        assert_eq!(lines[1].comment, None); // `@fact` dropped
        assert_eq!(lines[2].operands, "sp, sp, #16"); // `#16` isn't a comment
        assert_eq!(
            lines[3].comment.as_deref(),
            Some("=>This Inner Loop Header: Depth=1")
        );
    }

    #[test]
    fn labels_take_the_line_of_the_code_they_introduce() {
        let asm = "\
_main:
\t.loc\t1 4 0
\tsub\tsp, sp, #16
\t.loc\t1 7 0
\tb\tLBB0_2
LBB0_2:
\t.loc\t1 9 0
\tret
";
        let lines = parse(asm);
        assert_eq!(lines[0].kind, AsmKind::Function);
        assert_eq!(lines[0].source_line, Some(4));
        assert_eq!(lines[3].kind, AsmKind::Block);
        assert_eq!(lines[3].source_line, Some(9));
    }

    #[test]
    fn debug_sections_are_skipped_entirely() {
        let asm = "\
\tret
\t.section\t__TEXT,__cstring,cstring_literals
l_.str:                                 ; @.str
\t.asciz\t\"%d;x\\n\"
\t.section\t__DWARF,__debug_abbrev,regular,debug
Lsection_abbrev:
\t.byte\t1                               ; Abbreviation Code
";
        let lines = parse(asm);
        assert_eq!(
            kinds(&lines),
            vec![
                AsmKind::Instruction,
                AsmKind::Section,
                AsmKind::Data,
                AsmKind::Directive,
            ]
        );
        assert_eq!(lines[3].mnemonic, ".asciz");
        assert_eq!(lines[3].operands, "\"%d;x\\n\""); // `;` inside quotes kept
    }

    #[test]
    fn splits_comments_for_each_target_flavour() {
        assert_eq!(
            split_comment("\tmov\tw8, #42  ; =0x2a"),
            ("\tmov\tw8, #42  ", Some("=0x2a".to_string()))
        );
        assert_eq!(
            split_comment("\tmovl\t$1, %eax  ## imm = 0x1"),
            ("\tmovl\t$1, %eax  ", Some("imm = 0x1".to_string()))
        );
        assert_eq!(
            split_comment("\tmovl\t$1, %eax  # imm = 0x1").1.as_deref(),
            Some("imm = 0x1")
        );
        assert_eq!(
            split_comment("\tmov\tw8, #1  // =0x1").1.as_deref(),
            Some("=0x1")
        );
        assert_eq!(split_comment("\tadd\tx0, x0, #4").1, None);
    }

    #[test]
    fn elf_local_labels_are_recognised() {
        let asm = "\t.text\nmain:\n\tjmp\t.LBB0_1\n.LBB0_1:\n.Ltmp3:\n\tretq\n";
        let lines = parse(asm);
        assert_eq!(
            kinds(&lines),
            vec![
                AsmKind::Section,
                AsmKind::Function,
                AsmKind::Instruction,
                AsmKind::Block,
                AsmKind::Instruction,
            ]
        );
    }
}
