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
- This repo's CI has no trailer check yet, so nothing enforces the rules above: follow them by hand.

## Local checks

These are the commands `.github/workflows/ci.yml` and `image.yml` run. `protoc` must be installed (`build.rs` needs it).

```
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
node scripts/check-web-syntax.mjs
cargo audit
(cd proto && buf lint)
```

CI also runs `buf breaking` against main on PRs. The toolchain is pinned in `rust-toolchain.toml`.

Not runnable on a Mac: `cargo test` (the test binary fails to link ort-sys, "TRIGGER_BAD_DIST_FEATURES"; CI runs it on Linux), the CUDA image build (runs on the build-runner) and the `--selftest` GPU gate (needs the card; see README "Running the gate").

## Deploy

`image.yml` builds and releases on push to main. Never hand-edit `image.tag` in `helm/values.yaml`; the pipeline owns it.
