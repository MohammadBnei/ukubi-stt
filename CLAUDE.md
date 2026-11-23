# ukubi-stt

Rust service (OTel, gRPC). Change rules: mountain `plan/16-dev-rules.md`; code design: `plan/15-architecture.md`.

## Commits

- Run `git status` before every commit; never commit build output or binaries.
- Never push to main; work on a branch and open a PR (plan/16). `.github/workflows/trailers.yml` (ukubi-ci) checks Conventional titles/subjects and AI trailers.

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

`image.yml` builds and releases on push to main, unless the push touches only `*.md`, `helm/**` or `k8s/**`. Any other file, CI or scripts included, cuts at least a patch release and redeploys the single GPU pod (`strategy: Recreate`, so a short outage). Never hand-edit `image.tag`: the pipeline owns it, in `helm/image.yaml` on the `deploy` branch (ArgoCD layers it over `helm/values.yaml`).
ukubi-ci `release.yml` (git-cliff) cuts a bare annotated `X.Y.Z` tag (notes in its message, nothing committed to main) from the commit types, so the type is the bump. The image is tagged with the 12-char commit SHA and carries the version as the `VERSION` build-arg (OCI label, the `ukubi-stt starting` log line, `--version`).
