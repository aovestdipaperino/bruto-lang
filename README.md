# bruto-lang

[![GitHub stars](https://img.shields.io/github/stars/aovestdipaperino/bruto-lang?logo=github&color=brightgreen)](https://github.com/aovestdipaperino/bruto-lang/stargazers)

Language interfaces and runtime for the Bruto IDE, the Rust crate that [bruto-pascal](https://github.com/aovestdipaperino/bruto-pascal) builds on.

## Modules

| Module | Purpose |
|---|---|
| `language` | Language interface definitions |
| `runtime` | Runtime support |
| `target` | Compilation targets |
| `profile` | Build profiles |
| `disasm` | Disassembly support |

It builds on [`inkwell`](https://crates.io/crates/inkwell) (LLVM 18) and [`turbo-vision`](https://crates.io/crates/turbo-vision), so a matching LLVM 18 installation is required to build it.

## License

MIT
