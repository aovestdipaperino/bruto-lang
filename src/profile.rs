//! Language-agnostic profiling results, produced by an instrumented run
//! and consumed by the IDE. See docs/superpowers/specs/2026-09-17-line-profiler-design.md.

use std::collections::HashMap;

/// Magic bytes at the start of a `.bruto-prof` file.
pub const MAGIC: &[u8; 4] = b"BPRF";
/// File format version this reader understands.
pub const VERSION: u32 = 1;
const HEADER_LEN: usize = 24;
const NODE_LEN: usize = 33;
const FLAG_TABLE_FULL: u32 = 1;
const FLAG_STACK_OVERFLOW: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileKind {
    Routine,
    Line,
}

/// One node of the call tree: a routine invocation context or a source
/// line executed within one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileNode {
    pub kind: ProfileKind,
    /// Routine name; empty for lines.
    pub name: String,
    /// 1-based source line (the header line for routines).
    pub line: usize,
    /// Index into `Profile::nodes`; `None` for the root's children.
    pub parent: Option<usize>,
    pub calls: u64,
    pub self_ns: u64,
    pub total_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Profile {
    /// Wall-clock nanoseconds from the first hook to exit.
    pub elapsed_ns: u64,
    /// True when the runtime dropped data (node table full or stack too deep).
    pub truncated: bool,
    pub nodes: Vec<ProfileNode>,
}

/// One entry of the id map written by codegen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MapEntry {
    pub kind: ProfileKind,
    pub line: usize,
    pub name: String,
}

/// Location id -> source position, from `<exe>.bruto-prof-map`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProfMap {
    pub entries: HashMap<u32, MapEntry>,
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[at..at + 8]);
    u64::from_le_bytes(a)
}

impl Profile {
    /// Parse the map file: lines `P <id> <line> <name>` for routines and
    /// `L <id> <line> <col>` for statements. Blank lines are ignored.
    pub fn read_map(text: &str) -> Result<ProfMap, String> {
        let mut map = ProfMap::default();
        for (n, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() {
                continue;
            }
            let mut parts = line.splitn(4, ' ');
            let kind = parts.next().unwrap_or("");
            let id: u32 = parts
                .next()
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| format!("prof map line {}: bad id", n + 1))?;
            let src_line: usize = parts
                .next()
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| format!("prof map line {}: bad line number", n + 1))?;
            let rest = parts.next().unwrap_or("");
            let entry = match kind {
                "P" => MapEntry {
                    kind: ProfileKind::Routine,
                    line: src_line,
                    name: rest.to_string(),
                },
                "L" => MapEntry {
                    kind: ProfileKind::Line,
                    line: src_line,
                    name: String::new(),
                },
                other => return Err(format!("prof map line {}: unknown kind {other:?}", n + 1)),
            };
            map.entries.insert(id, entry);
        }
        Ok(map)
    }

    /// Decode a `.bruto-prof` file body, resolving ids through `map`.
    pub fn from_bytes(bytes: &[u8], map: &ProfMap) -> Result<Profile, String> {
        if bytes.len() < HEADER_LEN {
            return Err("profile file too short".into());
        }
        if &bytes[0..4] != MAGIC {
            return Err("profile file: bad magic".into());
        }
        let version = u32_at(bytes, 4);
        if version != VERSION {
            return Err(format!("profile file: unsupported version {version}"));
        }
        let flags = u32_at(bytes, 8);
        let elapsed_ns = u64_at(bytes, 12);
        let count = u32_at(bytes, 20) as usize;
        let need = HEADER_LEN + count * NODE_LEN;
        if bytes.len() < need {
            return Err(format!(
                "profile file: expected {need} bytes, got {}",
                bytes.len()
            ));
        }
        let mut nodes = Vec::with_capacity(count);
        for i in 0..count {
            let at = HEADER_LEN + i * NODE_LEN;
            let kind_byte = bytes[at];
            let loc = u32_at(bytes, at + 1);
            let parent_raw = u32_at(bytes, at + 5);
            let calls = u64_at(bytes, at + 9);
            let self_ns = u64_at(bytes, at + 17);
            let total_ns = u64_at(bytes, at + 25);
            let entry = map
                .entries
                .get(&loc)
                .ok_or_else(|| format!("profile file: unknown location id {loc}"))?;
            let kind = match kind_byte {
                1 => ProfileKind::Routine,
                2 => ProfileKind::Line,
                k => return Err(format!("profile file: bad node kind {k}")),
            };
            if kind != entry.kind {
                return Err(format!(
                    "profile file: node {i} kind does not match map entry {loc}"
                ));
            }
            let parent = if parent_raw == u32::MAX {
                None
            } else if (parent_raw as usize) < count {
                Some(parent_raw as usize)
            } else {
                return Err(format!("profile file: parent {parent_raw} out of range"));
            };
            nodes.push(ProfileNode {
                kind,
                name: entry.name.clone(),
                line: entry.line,
                parent,
                calls,
                self_ns,
                total_ns,
            });
        }
        for i in 0..nodes.len() {
            let mut cur = i;
            let mut steps = 0usize;
            loop {
                match nodes[cur].parent {
                    None => break,
                    Some(p) => {
                        if p == cur {
                            return Err(format!("profile file: cyclic parent chain at node {i}"));
                        }
                        cur = p;
                        steps += 1;
                        if steps > count {
                            return Err(format!("profile file: cyclic parent chain at node {i}"));
                        }
                    }
                }
            }
        }
        Ok(Profile {
            elapsed_ns,
            truncated: flags & (FLAG_TABLE_FULL | FLAG_STACK_OVERFLOW) != 0,
            nodes,
        })
    }

    /// Read both files from disk.
    pub fn load(profile_path: &str, map_path: &str) -> Result<Profile, String> {
        let map_text =
            std::fs::read_to_string(map_path).map_err(|e| format!("reading {map_path}: {e}"))?;
        let map = Self::read_map(&map_text)?;
        let bytes =
            std::fs::read(profile_path).map_err(|e| format!("reading {profile_path}: {e}"))?;
        Self::from_bytes(&bytes, &map)
    }

    /// Per-line `(self_ns, hits)` summed over every calling context.
    /// This is what the editor's heat column shows.
    pub fn line_totals(&self) -> HashMap<usize, (u64, u64)> {
        let mut out: HashMap<usize, (u64, u64)> = HashMap::new();
        for n in &self.nodes {
            if n.kind == ProfileKind::Line {
                let e = out.entry(n.line).or_insert((0, 0));
                e.0 += n.self_ns;
                e.1 += n.calls;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn le_u32(v: &mut Vec<u8>, x: u32) {
        v.extend_from_slice(&x.to_le_bytes());
    }
    fn le_u64(v: &mut Vec<u8>, x: u64) {
        v.extend_from_slice(&x.to_le_bytes());
    }

    fn node(
        v: &mut Vec<u8>,
        kind: u8,
        loc: u32,
        parent: u32,
        calls: u64,
        self_ns: u64,
        total_ns: u64,
    ) {
        v.push(kind);
        le_u32(v, loc);
        le_u32(v, parent);
        le_u64(v, calls);
        le_u64(v, self_ns);
        le_u64(v, total_ns);
    }

    fn sample_map() -> ProfMap {
        Profile::read_map("P 1 3 program\nL 2 5 3\nL 3 7 5\nP 4 10 Double\nL 5 12 3\n").unwrap()
    }

    fn sample_bytes(flags: u32) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(b"BPRF");
        le_u32(&mut v, 1);
        le_u32(&mut v, flags);
        le_u64(&mut v, 1_000);
        le_u32(&mut v, 4);
        node(&mut v, 1, 1, u32::MAX, 1, 100, 1_000); // program
        node(&mut v, 2, 2, 0, 5, 300, 300); // line 5 under program
        node(&mut v, 1, 4, 0, 5, 500, 600); // Double under program
        node(&mut v, 2, 5, 2, 5, 600, 600); // line 12 under Double
        v
    }

    #[test]
    fn parses_header_and_nodes() {
        let p = Profile::from_bytes(&sample_bytes(0), &sample_map()).unwrap();
        assert_eq!(p.elapsed_ns, 1_000);
        assert!(!p.truncated);
        assert_eq!(p.nodes.len(), 4);
        assert_eq!(p.nodes[0].kind, ProfileKind::Routine);
        assert_eq!(p.nodes[0].name, "program");
        assert_eq!(p.nodes[0].line, 3);
        assert_eq!(p.nodes[0].parent, None);
        assert_eq!(p.nodes[1].kind, ProfileKind::Line);
        assert_eq!(p.nodes[1].line, 5);
        assert_eq!(p.nodes[1].parent, Some(0));
        assert_eq!(p.nodes[3].parent, Some(2));
        assert_eq!(p.nodes[2].calls, 5);
    }

    #[test]
    fn truncated_flags_are_reported() {
        assert!(
            Profile::from_bytes(&sample_bytes(1), &sample_map())
                .unwrap()
                .truncated
        );
        assert!(
            Profile::from_bytes(&sample_bytes(2), &sample_map())
                .unwrap()
                .truncated
        );
    }

    #[test]
    fn rejects_bad_magic_and_short_data() {
        let mut bad = sample_bytes(0);
        bad[0] = b'X';
        assert!(Profile::from_bytes(&bad, &sample_map()).is_err());
        let short = &sample_bytes(0)[..30];
        assert!(Profile::from_bytes(short, &sample_map()).is_err());
    }

    #[test]
    fn unknown_location_id_is_an_error() {
        let map = Profile::read_map("P 1 3 program\n").unwrap();
        assert!(Profile::from_bytes(&sample_bytes(0), &map).is_err());
    }

    #[test]
    fn line_totals_aggregate_across_callers() {
        let p = Profile::from_bytes(&sample_bytes(0), &sample_map()).unwrap();
        let t = p.line_totals();
        assert_eq!(t.get(&5), Some(&(300, 5)));
        assert_eq!(t.get(&12), Some(&(600, 5)));
        assert_eq!(t.get(&3), None, "routine rows are not lines");
    }

    #[test]
    fn rejects_cyclic_parent_chain() {
        let mut v = Vec::new();
        v.extend_from_slice(b"BPRF");
        le_u32(&mut v, 1);
        le_u32(&mut v, 0);
        le_u64(&mut v, 1_000);
        le_u32(&mut v, 2);
        node(&mut v, 1, 1, 1, 1, 100, 1_000); // node 0, parent 1
        node(&mut v, 1, 4, 0, 1, 100, 1_000); // node 1, parent 0
        let err = Profile::from_bytes(&v, &sample_map()).unwrap_err();
        assert!(err.contains("cyclic parent chain"), "{err}");
    }

    #[test]
    fn rejects_self_parent() {
        let mut v = Vec::new();
        v.extend_from_slice(b"BPRF");
        le_u32(&mut v, 1);
        le_u32(&mut v, 0);
        le_u64(&mut v, 1_000);
        le_u32(&mut v, 1);
        node(&mut v, 1, 1, 0, 1, 100, 1_000); // node 0, parent itself
        let err = Profile::from_bytes(&v, &sample_map()).unwrap_err();
        assert!(err.contains("cyclic parent chain"), "{err}");
    }

    #[test]
    fn rejects_kind_mismatch() {
        let mut v = Vec::new();
        v.extend_from_slice(b"BPRF");
        le_u32(&mut v, 1);
        le_u32(&mut v, 0);
        le_u64(&mut v, 1_000);
        le_u32(&mut v, 1);
        node(&mut v, 2, 1, u32::MAX, 1, 100, 1_000); // kind 2 (Line) but map id 1 is "P"
        let err = Profile::from_bytes(&v, &sample_map()).unwrap_err();
        assert!(err.contains("kind does not match map entry"), "{err}");
    }

    #[test]
    fn map_parser_skips_blank_and_rejects_garbage() {
        let m = Profile::read_map("\nL 7 9 1\n\n").unwrap();
        assert_eq!(m.entries[&7].line, 9);
        assert!(Profile::read_map("Q 1 2 3\n").is_err());
        assert!(Profile::read_map("L x 2 3\n").is_err());
    }
}
