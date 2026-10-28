//! The one place configuration comes from: the arguments and the environment,
//! read once at start, validated together, every error reported at once.
//!
//! [`Config::load`] takes both as plain iterators so tests pass a list.
//! [`Config::from_process`] is the only code in the crate that reads the real
//! process; `clippy.toml` bans every other environment reader, and
//! `tests::lint_escapes_are_only_at_the_approved_sites` holds the exemptions to a
//! fixed list. Every variable and flag is in the README's `### Configuration`
//! tables, and `tests::readme_tables_match_what_the_loader_reads` holds the two
//! to the same set.

use crate::persian::Device;
use std::{collections::BTreeSet, ffi::OsString, fmt, path::PathBuf};

/// Default cap on concurrent streaming sessions (`STT_MAX_SESSIONS`).
///
/// The binding constraint is not the ~7.5MB of decoder state each one holds —
/// it is the GPU, which each stream occupies for 20-50ms out of every 560ms.
/// Eight is roughly half the card's duty cycle with the batch path still
/// needing room, but that is an estimate, and the right number is whatever the
/// card sustains once real consumers are on it. Configurable so tuning it is a
/// values edit rather than a rebuild.
pub const DEFAULT_MAX_SESSIONS: usize = 8;

const NO_CREDENTIALS: &str = "no credentials configured. Set STT_AUTH_TOKEN, or one \
     STT_TOKEN_<NAME> per caller, via Infisical (project ukubi-stt-bhr-m). Refusing to \
     serve a GPU with no authentication on a publicly-resolvable hostname.";

#[derive(Debug)]
pub struct Config {
    pub mode: Mode,
    pub model_dir: PathBuf,
    pub stream_dir: PathBuf,
    pub fa_dir: PathBuf,
    /// Validated in every mode, used only by `--selftest-fa`.
    pub fa_device: Device,
    pub max_sessions: usize,
    /// `RUST_LOG` verbatim. Unset is `None`; empty is `Some("")`, which
    /// tracing-subscriber parses as "no directives" rather than falling back.
    pub log_filter: Option<String>,
}

#[derive(Debug)]
pub enum Mode {
    /// No argument. The only mode that needs a credential.
    Serve {
        credentials: Credentials,
    },
    Selftest {
        wav: Option<PathBuf>,
    },
    SelftestStream,
    /// An empty `reference` stays `Some("")`: gate-fa-pod passes `"$REF"`.
    SelftestFa {
        wav: Option<PathBuf>,
        reference: Option<String>,
    },
}

/// `(name, token)` per caller, sorted by name, names unique and non-empty.
/// `Debug` prints names only, so deriving `Debug` on anything that holds one
/// cannot leak a token. The field is private: only `Config::load` builds one,
/// so a `Credentials` is never empty.
pub struct Credentials(Vec<(String, Vec<u8>)>);

impl Credentials {
    pub fn names(&self) -> Vec<&str> {
        self.0.iter().map(|(n, _)| n.as_str()).collect()
    }

    pub fn into_vec(self) -> Vec<(String, Vec<u8>)> {
        self.0
    }
}

impl fmt::Debug for Credentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.names()).finish()
    }
}

impl Config {
    #[expect(
        clippy::disallowed_methods,
        reason = "the one reader of the real process"
    )]
    pub fn from_process() -> anyhow::Result<Self> {
        Self::load(std::env::args_os().skip(1), std::env::vars_os())
    }

    /// `args` excludes the program name.
    pub fn load(
        args: impl IntoIterator<Item = OsString>,
        env: impl IntoIterator<Item = (OsString, OsString)>,
    ) -> anyhow::Result<Self> {
        load_recorded(args, env).0
    }
}

/// What the loader looked at, so the README test can hold the tables to it.
#[cfg_attr(not(test), allow(dead_code, reason = "only the README test reads it"))]
struct Reads {
    keys: BTreeSet<String>,
    flags: BTreeSet<&'static str>,
}

/// The environment, as the loader is allowed to see it. Every lookup goes
/// through `get` or `with_prefix`, which record the key — that is what makes
/// the README test mean something.
struct Env {
    vars: Vec<(String, String)>,
    read: BTreeSet<String>,
}

impl Env {
    /// Keys this service owns. A non-UTF-8 value under one is refused; anything
    /// else is somebody else's (a library's, the shell's) and is ignored.
    fn owned(key: &str) -> bool {
        key.starts_with("STT_") || key == "RUST_LOG"
    }

    fn new(raw: impl IntoIterator<Item = (OsString, OsString)>, errors: &mut Vec<String>) -> Self {
        let mut vars = Vec::new();
        for (k, v) in raw {
            match (k.into_string(), v.into_string()) {
                (Ok(k), Ok(v)) if Self::owned(&k) => vars.push((k, v)),
                // Names the key, never the value: it may be a token.
                (Ok(k), Err(_)) if Self::owned(&k) => {
                    errors.push(format!("{k} is not valid UTF-8"))
                }
                (Err(k), _) if Self::owned(&k.to_string_lossy()) => errors.push(format!(
                    "the variable name {:?} is not valid UTF-8",
                    k.to_string_lossy()
                )),
                _ => {}
            }
        }
        Self {
            vars,
            read: BTreeSet::new(),
        }
    }

    /// Unset and empty stay apart: `None` against `Some("")`.
    fn get(&mut self, key: &str) -> Option<String> {
        self.read.insert(key.to_string());
        self.vars
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.clone())
    }

    /// `(suffix, value)` for every key starting with `prefix`, in environment
    /// order. Recorded as `<prefix><NAME>`, which is how the README spells it.
    fn with_prefix(&mut self, prefix: &str) -> Vec<(String, String)> {
        self.read.insert(format!("{prefix}<NAME>"));
        self.vars
            .iter()
            .filter_map(|(k, v)| Some((k.strip_prefix(prefix)?.to_string(), v.clone())))
            .collect()
    }
}

/// Unset or empty is the default. Anything else that is not a positive integer
/// refuses: zero would silently disable streaming, and a typo that quietly
/// becomes 8 is a value nobody chose.
fn parse_max_sessions(raw: Option<&str>) -> Result<usize, String> {
    match raw {
        None | Some("") => Ok(DEFAULT_MAX_SESSIONS),
        Some(raw) => match raw.parse::<usize>() {
            Ok(n) if n > 0 => Ok(n),
            _ => Err(format!(
                "STT_MAX_SESSIONS={raw:?} is not a positive integer"
            )),
        },
    }
}

fn load_recorded(
    args: impl IntoIterator<Item = OsString>,
    env: impl IntoIterator<Item = (OsString, OsString)>,
) -> (anyhow::Result<Config>, Reads) {
    let mut errors = Vec::new();
    let mut env = Env::new(env, &mut errors);

    // Trailing extra arguments are ignored, as they always have been. A
    // non-UTF-8 one is refused rather than panicking, as args() would.
    let mut args = args.into_iter().filter_map(|a| {
        a.into_string()
            .map_err(|a| errors.push(format!("the argument {a:?} is not valid UTF-8")))
            .ok()
    });
    let first = args.next();
    let mut flags = BTreeSet::new();
    let mut is = |flag: &'static str| {
        flags.insert(flag);
        first.as_deref() == Some(flag)
    };
    enum Want {
        Serve,
        Selftest(Option<PathBuf>),
        Stream,
        Fa(Option<PathBuf>, Option<String>),
    }
    let want = if first.is_none() {
        Some(Want::Serve)
    } else if is("--selftest") {
        Some(Want::Selftest(args.next().map(PathBuf::from)))
    } else if is("--selftest-stream") {
        Some(Want::Stream)
    } else if is("--selftest-fa") {
        let wav = args.next().map(PathBuf::from);
        Some(Want::Fa(wav, args.next()))
    } else {
        errors.push(format!(
            "unknown argument {:?} — expected --selftest [wav], --selftest-stream, \
             or --selftest-fa [wav] [reference text]",
            first.as_deref().unwrap_or_default()
        ));
        None
    };

    // `unwrap_or_else` on an Option, not `env::var(..).unwrap_or_else`: an
    // empty value is kept as an empty path, exactly as before.
    let dir = |env: &mut Env, key: &str, default: &str| {
        PathBuf::from(env.get(key).unwrap_or_else(|| default.to_string()))
    };
    let model_dir = dir(&mut env, "STT_MODEL_DIR", "/models/tdt");
    let stream_dir = dir(&mut env, "STT_STREAM_MODEL_DIR", "/models/nemotron");
    let fa_dir = dir(&mut env, "STT_FA_MODEL_DIR", "/models/shenava");

    let max_sessions =
        parse_max_sessions(env.get("STT_MAX_SESSIONS").as_deref()).unwrap_or_else(|e| {
            errors.push(e);
            DEFAULT_MAX_SESSIONS
        });

    let fa_device = match env.get("STT_FA_DEVICE").as_deref() {
        Some("cuda") => Device::Cuda,
        Some("cpu") | None => Device::Cpu,
        Some(other) => {
            errors.push(format!(
                "STT_FA_DEVICE={other:?} — expected \"cpu\" or \"cuda\""
            ));
            Device::Cpu
        }
    };

    let log_filter = env.get("RUST_LOG");

    // One credential per caller as STT_TOKEN_<NAME>, plus the original
    // STT_AUTH_TOKEN as `default`. Empty values are not credentials.
    let mut clients: Vec<(String, Vec<u8>)> = env
        .with_prefix("STT_TOKEN_")
        .into_iter()
        .map(|(name, v)| (name.to_lowercase(), v))
        .chain(
            env.get("STT_AUTH_TOKEN")
                .map(|v| ("default".to_string(), v)),
        )
        .filter(|(_, v)| !v.is_empty())
        .map(|(n, v)| (n, v.into_bytes()))
        .collect();
    clients.sort_by(|a, b| a.0.cmp(&b.0));
    // The name is what keeps one caller out of another's session_id, so two
    // credentials under one name (case, or STT_TOKEN_DEFAULT beside
    // STT_AUTH_TOKEN) refuse. Names the client, never the token.
    if clients.iter().any(|(n, _)| n.is_empty()) {
        errors.push("STT_TOKEN_ has no client name: use STT_TOKEN_<NAME>".to_string());
    }
    for pair in clients.windows(2) {
        if pair[0].0 == pair[1].0 && !pair[0].0.is_empty() {
            errors.push(format!(
                "two credentials for the client {:?}: names are case-insensitive, \
                 and STT_AUTH_TOKEN is \"default\"",
                pair[0].0
            ));
        }
    }

    let mode = want.map(|w| match w {
        Want::Serve => {
            if clients.is_empty() {
                errors.push(NO_CREDENTIALS.to_string());
            }
            Mode::Serve {
                credentials: Credentials(clients),
            }
        }
        Want::Selftest(wav) => Mode::Selftest { wav },
        Want::Stream => Mode::SelftestStream,
        Want::Fa(wav, reference) => Mode::SelftestFa { wav, reference },
    });

    let reads = Reads {
        keys: env.read,
        flags,
    };
    let result = match mode {
        Some(mode) if errors.is_empty() => Ok(Config {
            mode,
            model_dir,
            stream_dir,
            fa_dir,
            fa_device,
            max_sessions,
            log_filter,
        }),
        _ => Err(anyhow::anyhow!(errors.join("\n"))),
    };
    (result, reads)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> Vec<(OsString, OsString)> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).into(), (*v).into()))
            .collect()
    }

    fn load(args: &[&str], pairs: &[(&str, &str)]) -> anyhow::Result<Config> {
        Config::load(args.iter().map(OsString::from), env(pairs))
    }

    fn err(args: &[&str], pairs: &[(&str, &str)]) -> Vec<String> {
        load(args, pairs)
            .unwrap_err()
            .to_string()
            .lines()
            .map(String::from)
            .collect()
    }

    const T: (&str, &str) = ("STT_AUTH_TOKEN", "a");

    #[test]
    fn defaults_when_unset() {
        let c = load(&[], &[T]).unwrap();
        assert_eq!(c.model_dir, PathBuf::from("/models/tdt"));
        assert_eq!(c.stream_dir, PathBuf::from("/models/nemotron"));
        assert_eq!(c.fa_dir, PathBuf::from("/models/shenava"));
        assert_eq!(c.fa_device, Device::Cpu);
        assert_eq!(c.max_sessions, DEFAULT_MAX_SESSIONS);
        assert_eq!(c.log_filter, None);
        assert!(matches!(c.mode, Mode::Serve { .. }));
    }

    #[test]
    fn set_values_land_in_their_own_fields() {
        let c = load(
            &[],
            &[
                T,
                ("STT_MODEL_DIR", "/a"),
                ("STT_STREAM_MODEL_DIR", "/b"),
                ("STT_FA_MODEL_DIR", "/c"),
                ("STT_FA_DEVICE", "cuda"),
                ("STT_MAX_SESSIONS", "12"),
                ("RUST_LOG", "info,ort=info"),
            ],
        )
        .unwrap();
        assert_eq!(c.model_dir, PathBuf::from("/a"));
        assert_eq!(c.stream_dir, PathBuf::from("/b"));
        assert_eq!(c.fa_dir, PathBuf::from("/c"));
        assert_eq!(c.fa_device, Device::Cuda);
        assert_eq!(c.max_sessions, 12);
        assert_eq!(c.log_filter.as_deref(), Some("info,ort=info"));
    }

    /// Empty is not unset, for every variable, as before this loader existed.
    #[test]
    fn empty_is_not_unset() {
        let c = load(
            &[],
            &[
                T,
                ("STT_MODEL_DIR", ""),
                ("STT_STREAM_MODEL_DIR", ""),
                ("STT_FA_MODEL_DIR", ""),
                ("STT_MAX_SESSIONS", ""),
                ("RUST_LOG", ""),
            ],
        )
        .unwrap();
        assert_eq!(c.model_dir, PathBuf::from(""));
        assert_eq!(c.stream_dir, PathBuf::from(""));
        assert_eq!(c.fa_dir, PathBuf::from(""));
        assert_eq!(c.max_sessions, DEFAULT_MAX_SESSIONS);
        assert_eq!(c.log_filter.as_deref(), Some(""));
        assert_eq!(
            err(&["--selftest-stream"], &[("STT_FA_DEVICE", "")]),
            [r#"STT_FA_DEVICE="" — expected "cpu" or "cuda""#]
        );
    }

    #[test]
    fn every_error_at_once() {
        let bad = [("STT_MAX_SESSIONS", "lots"), ("STT_FA_DEVICE", "bogus")];
        assert_eq!(
            err(&["--bogus"], &bad),
            [
                "unknown argument \"--bogus\" — expected --selftest [wav], --selftest-stream, \
                 or --selftest-fa [wav] [reference text]",
                r#"STT_MAX_SESSIONS="lots" is not a positive integer"#,
                r#"STT_FA_DEVICE="bogus" — expected "cpu" or "cuda""#,
            ]
        );
        assert_eq!(
            err(&[], &bad),
            [
                r#"STT_MAX_SESSIONS="lots" is not a positive integer"#,
                r#"STT_FA_DEVICE="bogus" — expected "cpu" or "cuda""#,
                NO_CREDENTIALS,
            ]
        );
    }

    #[test]
    fn credentials_not_required_outside_serve() {
        assert_eq!(err(&[], &[]), [NO_CREDENTIALS]);
        assert_eq!(err(&[], &[("STT_AUTH_TOKEN", "")]), [NO_CREDENTIALS]);
        for args in [
            &["--selftest"][..],
            &["--selftest-stream"],
            &["--selftest-fa"],
        ] {
            load(args, &[]).unwrap();
        }
    }

    #[test]
    fn token_naming() {
        let c = load(
            &[],
            &[
                ("STT_TOKEN_FLEET", "b"),
                ("STT_AUTH_TOKEN", "a"),
                ("STT_TOKEN_X", ""),
                ("STT_TOKEN_Dreamer", "c"),
                ("OTHER_TOKEN", "z"),
            ],
        )
        .unwrap();
        let Mode::Serve { credentials } = c.mode else {
            panic!("not serve")
        };
        assert_eq!(
            credentials.0,
            [
                ("default".to_string(), b"a".to_vec()),
                ("dreamer".to_string(), b"c".to_vec()),
                ("fleet".to_string(), b"b".to_vec()),
            ]
        );
        assert_eq!(
            format!("{credentials:?}"),
            r#"["default", "dreamer", "fleet"]"#
        );
    }

    #[test]
    fn clashing_or_empty_client_names_refuse() {
        let clash = "two credentials for the client \"default\": names are case-insensitive, \
                     and STT_AUTH_TOKEN is \"default\"";
        assert_eq!(err(&[], &[T, ("STT_TOKEN_DEFAULT", "b")]), [clash]);
        assert_eq!(
            err(&[], &[("STT_TOKEN_Fleet", "a"), ("STT_TOKEN_FLEET", "b")]),
            [clash.replace("default\":", "fleet\":")]
        );
        assert_eq!(
            err(&[], &[T, ("STT_TOKEN_", "b")]),
            ["STT_TOKEN_ has no client name: use STT_TOKEN_<NAME>"]
        );
    }

    #[test]
    fn non_utf8_argument_refuses() {
        use std::os::unix::ffi::OsStringExt;
        let args = vec!["--selftest".into(), OsString::from_vec(vec![0xff])];
        let e = Config::load(args, env(&[])).unwrap_err().to_string();
        assert_eq!(e, "the argument \"\\xFF\" is not valid UTF-8");
    }

    #[test]
    fn max_sessions_refuses_nonsense() {
        assert_eq!(parse_max_sessions(None), Ok(DEFAULT_MAX_SESSIONS));
        assert_eq!(parse_max_sessions(Some("")), Ok(DEFAULT_MAX_SESSIONS));
        assert_eq!(parse_max_sessions(Some("12")), Ok(12));

        // A zero cap would disable streaming entirely and report it as
        // RESOURCE_EXHAUSTED on the first chunk — indistinguishable from a busy
        // GPU, which is exactly the wrong way to learn about a typo.
        for bad in ["0", "-3", "lots", " 12"] {
            assert_eq!(
                parse_max_sessions(Some(bad)),
                Err(format!(
                    "STT_MAX_SESSIONS={bad:?} is not a positive integer"
                ))
            );
        }
    }

    #[test]
    fn arguments_keep_their_edges() {
        // Extras are ignored.
        let c = load(&["--selftest", "a.wav", "junk"], &[]).unwrap();
        assert!(
            matches!(c.mode, Mode::Selftest { wav: Some(w) } if w == std::path::Path::new("a.wav"))
        );
        assert!(matches!(
            load(&["--selftest-stream", "x"], &[]).unwrap().mode,
            Mode::SelftestStream
        ));
        // An empty reference is a reference, not an absent one.
        let c = load(&["--selftest-fa", "/a.wav", ""], &[]).unwrap();
        assert!(
            matches!(c.mode, Mode::SelftestFa { wav: Some(_), reference: Some(r) } if r.is_empty())
        );
        let c = load(&["--selftest-fa"], &[]).unwrap();
        assert!(matches!(
            c.mode,
            Mode::SelftestFa {
                wav: None,
                reference: None
            }
        ));
        assert_eq!(
            err(&[""], &[]),
            [
                "unknown argument \"\" — expected --selftest [wav], --selftest-stream, \
              or --selftest-fa [wav] [reference text]"
            ]
        );
    }

    #[test]
    fn non_utf8_owned_value_refuses() {
        use std::os::unix::ffi::OsStringExt;
        let bad = || OsString::from_vec(vec![0xff]);
        let with = |k: &str| {
            let mut e = env(&[T]);
            e.push((k.into(), bad()));
            Config::load(Vec::new(), e)
        };
        for key in [
            "STT_AUTH_TOKEN",
            "STT_TOKEN_FLEET",
            "STT_MODEL_DIR",
            "RUST_LOG",
        ] {
            let e = with(key).unwrap_err().to_string();
            assert_eq!(e, format!("{key} is not valid UTF-8"));
        }
        // Not ours: ignored, where std::env::vars() used to panic on it.
        with("FOO").unwrap();
        let mut e = env(&[T]);
        e.push((OsString::from_vec(b"STT_TOKEN_\xff".to_vec()), "x".into()));
        assert!(Config::load(Vec::new(), e)
            .unwrap_err()
            .to_string()
            .contains("is not valid UTF-8"));
    }

    /// First cells of the table rows under `### Configuration`, split into
    /// flags (`--…`) and variables. Prose and the library notes are not rows.
    // include_str! stays test-only: the image build copies no README.
    fn readme_rows() -> (BTreeSet<String>, BTreeSet<String>) {
        let readme = include_str!("../README.md");
        let section = readme
            .split("\n### Configuration\n")
            .nth(1)
            .expect("README has a ### Configuration section");
        let section = section.split("\n## ").next().unwrap();
        let section = section.split("\n### ").next().unwrap();
        let (mut flags, mut keys) = (BTreeSet::new(), BTreeSet::new());
        for line in section.lines() {
            let Some(cell) = line.strip_prefix("| `") else {
                continue;
            };
            let first = cell.split('`').next().unwrap();
            let name = first.split_whitespace().next().unwrap().to_string();
            if name.starts_with("--") {
                flags.insert(name);
            } else {
                keys.insert(name);
            }
        }
        (flags, keys)
    }

    #[test]
    fn readme_tables_match_what_the_loader_reads() {
        // An unknown argument makes the parser compare against every flag.
        let (_, reads) = load_recorded(vec!["--not-a-flag".into()], env(&[]));
        assert!(reads.keys.contains("STT_MODEL_DIR"), "positive control");
        assert!(reads.flags.contains("--selftest-fa"), "positive control");
        let (flags, keys) = readme_rows();
        let read_flags: BTreeSet<String> = reads.flags.iter().map(|s| s.to_string()).collect();
        assert_eq!(keys, reads.keys, "README variables vs loader reads");
        assert_eq!(flags, read_flags, "README flags vs parser flags");
    }

    /// Every `.rs` file under `src/`, recursively, plus build.rs. The count is
    /// pinned so a file the walk misses, or a new module, fails here first.
    fn rust_sources() -> Vec<PathBuf> {
        fn walk(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
            for e in std::fs::read_dir(dir).unwrap() {
                let p = e.unwrap().path();
                if p.is_dir() {
                    walk(&p, out);
                } else if p.extension().is_some_and(|x| x == "rs") {
                    out.push(p);
                }
            }
        }
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut files = vec![root.join("build.rs")];
        walk(&root.join("src"), &mut files);
        files.sort();
        let names: Vec<String> = files
            .iter()
            .map(|p| p.strip_prefix(root).unwrap().display().to_string())
            .collect();
        assert_eq!(
            names,
            [
                "build.rs",
                "src/config.rs",
                "src/engine.rs",
                "src/errkeys.rs",
                "src/fbank.rs",
                "src/main.rs",
                "src/persian.rs",
                "src/service.rs",
            ],
            "a new source file: add it here and re-check the counts below"
        );
        files
    }

    /// Counts `needle` per source file, keeping only the files that have it.
    fn count_in_sources(needle: &str) -> std::collections::BTreeMap<String, usize> {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        rust_sources()
            .iter()
            .filter_map(|p| {
                let n = std::fs::read_to_string(p).unwrap().matches(needle).count();
                let name = p.strip_prefix(root).unwrap().display().to_string();
                (n > 0).then_some((name, n))
            })
            .collect()
    }

    fn counts(pairs: &[(&str, usize)]) -> std::collections::BTreeMap<String, usize> {
        pairs.iter().map(|&(k, v)| (k.to_string(), v)).collect()
    }

    /// clippy.toml's ban is only as good as its exemptions. Any
    /// `allow`/`expect` of the lint, or of a group containing it, outside these
    /// sites is a second config reader.
    #[test]
    fn lint_escapes_are_only_at_the_approved_sites() {
        for path in rust_sources() {
            let text = std::fs::read_to_string(&path).unwrap();
            for group in [
                concat!("clippy", "::all"),
                concat!("clippy", "::style"),
                concat!("(warn", "ings"),
                concat!(", warn", "ings"),
            ] {
                assert!(!text.contains(group), "{} silences {group}", path.display());
            }
        }
        let manifest = include_str!("../Cargo.toml");
        assert!(
            !manifest.contains(concat!("[lin", "ts")),
            "Cargo.toml lint table"
        );
        // config.rs: from_process. engine.rs: gpu_used_mib (exec) and one test.
        // main.rs: its test module. build.rs: OUT_DIR at build time.
        // errkeys.rs: the one Status constructor, a single-expression fn, so the
        // same lint's env ban is lifted for nothing else; the std-env count in the
        // next test still holds errkeys.rs at zero.
        assert_eq!(
            count_in_sources(concat!("disallowed", "_methods")),
            counts(&[
                ("build.rs", 1),
                ("src/config.rs", 1),
                ("src/engine.rs", 2),
                ("src/errkeys.rs", 1),
                ("src/main.rs", 1),
            ])
        );
    }

    /// Log words are an interface: log-error-rate-spike counts ERROR lines. A new
    /// or removed ERROR site fails here until someone decides it should page:
    /// the macro, errkeys::internal (which logs one), and the Level spelling.
    #[test]
    fn the_error_sites_are_known() {
        assert_eq!(
            count_in_sources(concat!("tracing::", "error!")),
            counts(&[("src/errkeys.rs", 1), ("src/main.rs", 2)])
        );
        assert_eq!(
            count_in_sources(concat!("errkeys::", "internal(")),
            counts(&[("src/service.rs", 8)])
        );
        assert_eq!(count_in_sources(concat!("Level::", "ERROR")), counts(&[]));
    }

    /// The goal itself, without clippy: no file outside the exempt sites names
    /// `std::env` or spawns a process. A token count, so it also catches a
    /// reader clippy would only see under `-D warnings` in CI. config.rs's
    /// count includes two comments.
    #[test]
    fn only_the_exempt_sites_touch_the_environment() {
        assert_eq!(
            count_in_sources(concat!("env", "::")),
            counts(&[
                ("build.rs", 1),
                ("src/config.rs", 4),
                ("src/engine.rs", 1),
                ("src/main.rs", 2),
            ]),
            "a new std::env use: read it through Config instead"
        );
        assert_eq!(
            count_in_sources(concat!("Command", "::new")),
            counts(&[("src/engine.rs", 1), ("src/main.rs", 2)]),
            "a new child process inherits every credential"
        );
    }
}
