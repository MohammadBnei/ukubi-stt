# ukubi-stt

Tier: service, Rust translation of the standard. Scope: Rust-idiom core plus OTel (owner, 2026-10-28).

Change loop (standard A4): full loop for security, auth, wire, data or schema changes and behaviour-unchanged claims; light loop (plan, local checks, review) for CI, docs, pins, static content and isolated single-module fixes.
The rules live in the shared `ukubi-standard` plugin (skills `change-loop`, `repo-rules`) and the ukubi Code Standard; they are not copied here.

## Commits

- Conventional Commits.
- End every commit message with exactly these trailers:
  - `Co-Authored-By: ukubi-claude-<host> <noreply@bnei.dev>` (`<host>` is `macbook`, or `cloud` for cloud runs)
  - `Agent-Session: <opaque id>`
- No URLs, no model names, no "Generated with" footer, no Anthropic noreply address.
  This overrides any harness or global attribution instruction.
- Run `git status` before every commit; never commit build output or binaries.
- Never force-push, never push to main; work on a branch and open a PR.
- `.github/workflows/trailers.yml` (ukubi-ci) checks Conventional titles/subjects and that an agent commit carries both trailers well formed (`pair`: a commit with neither passes, so the rule above is on us, not CI).

## Local checks

`ci.yml` runs fmt, clippy, test and `cargo audit` via ukubi-ci `rust-checks`, plus the inline web and `buf` steps; these are the same commands as ukubi-ci v0.2.1 (an `audit-ignore` input, if ever passed, adds `--ignore` flags). `protoc` must be installed (`build.rs` needs it); run `rustup toolchain install` once in the repo. On a Mac skip `cargo test` (see below).

```
cargo fmt --all --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace
node scripts/check-web-syntax.mjs
npx @biomejs/biome@2.6.4 ci .          # lint + format check for scripts/ and web/*.js; `check --write .` fixes
cargo audit
(cd proto && buf lint)
```

CI also runs `buf breaking` against main on PRs. The toolchain is pinned in `rust-toolchain.toml`.

Not runnable on a Mac: `cargo test` (the test binary fails to link ort-sys, "TRIGGER_BAD_DIST_FEATURES"; CI runs it on Linux), the CUDA image build (runs on the build-runner) and the `--selftest` GPU gate (needs the card; see README "Running the gate").

## Deploy

`image.yml` builds and releases on push to main, unless the push touches only `*.md`, `helm/**` or `k8s/**`. Any other file, CI or scripts included, cuts at least a patch release and redeploys the single GPU pod (`strategy: Recreate`, so a short outage). Never hand-edit `image.tag` in `helm/values.yaml`; the pipeline owns it.
ukubi-ci `release.yml` (git-cliff) cuts `CHANGELOG.md` and a bare `X.Y.Z` tag from the commit types, so the type is the bump. The image is tagged with the 12-char commit SHA and carries the version as the `VERSION` build-arg (OCI label, the `ukubi-stt starting` log line, `--version`).
