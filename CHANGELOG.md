# Changelog

# [0.9.0](https://github.com/MohammadBnei/ukubi-stt/compare/0.8.0...0.9.0) (2026-09-01)


### Bug Fixes

* **release:** bump Cargo.lock too, or the next image build fails ([#19](https://github.com/MohammadBnei/ukubi-stt/issues/19)) ([91ea2f5](https://github.com/MohammadBnei/ukubi-stt/commit/91ea2f57ac668d3559077652d67f39f7cc688680))


### Features

* **stt:** NeMo log-mel for the Persian model, verified against the real model ([#17](https://github.com/MohammadBnei/ukubi-stt/issues/17)) ([778e329](https://github.com/MohammadBnei/ukubi-stt/commit/778e3293e0aa477c4623335a649203637151dc41))
* **stt:** the Persian ORT session, and --selftest-fa to gate it ([#18](https://github.com/MohammadBnei/ukubi-stt/issues/18)) ([2e10d2a](https://github.com/MohammadBnei/ukubi-stt/commit/2e10d2afdecf1298375cece26c732049d4311bd2))

# [0.8.0](https://github.com/MohammadBnei/ukubi-stt/compare/0.7.0...0.8.0) (2026-09-01)


### Features

* **web:** prewarm the audio graph so the first words are captured ([#15](https://github.com/MohammadBnei/ukubi-stt/issues/15)) ([ca81ca2](https://github.com/MohammadBnei/ukubi-stt/commit/ca81ca2d0622aec9166a265d424912d8af6bd2b3))

# [0.7.0](https://github.com/MohammadBnei/ukubi-stt/compare/0.6.0...0.7.0) (2026-09-01)


### Bug Fixes

* **streaming:** a close must never open a session ([#12](https://github.com/MohammadBnei/ukubi-stt/issues/12)) ([759c318](https://github.com/MohammadBnei/ukubi-stt/commit/759c318552ff5a4276f208035d9591efcd1164f5))


### Features

* STT_MAX_SESSIONS, so the streaming cap is tunable ([#13](https://github.com/MohammadBnei/ukubi-stt/issues/13)) ([4e21ebd](https://github.com/MohammadBnei/ukubi-stt/commit/4e21ebd6a9c1dbe65e0b608d710ea9dff17350be))
* **web:** extract the capture module, served as one shared copy ([#14](https://github.com/MohammadBnei/ukubi-stt/issues/14)) ([6ecb1f0](https://github.com/MohammadBnei/ukubi-stt/commit/6ecb1f03edf6ba17129ed678f368767e4f2ba6d8))

# [0.6.0](https://github.com/MohammadBnei/ukubi-stt/compare/0.5.3...0.6.0) (2026-08-31)


### Features

* gRPC reflection and per-client bearer tokens ([#11](https://github.com/MohammadBnei/ukubi-stt/issues/11)) ([0ddb148](https://github.com/MohammadBnei/ukubi-stt/commit/0ddb148b2a82e3bbe0ea38599f7d3135877ebf82))

## [0.5.3](https://github.com/MohammadBnei/ukubi-stt/compare/0.5.2...0.5.3) (2026-08-31)


### Performance Improvements

* **web:** AudioWorklet instead of ScriptProcessor, and chunk in the worklet ([#10](https://github.com/MohammadBnei/ukubi-stt/issues/10)) ([78147cc](https://github.com/MohammadBnei/ukubi-stt/commit/78147cce3562e7f92f7e910fbce8d084bf2a37c5))

## [0.5.2](https://github.com/MohammadBnei/ukubi-stt/compare/0.5.1...0.5.2) (2026-08-31)


### Bug Fixes

* **streaming:** chunks were 768ms, not 560ms — and the tail waited a round ([#9](https://github.com/MohammadBnei/ukubi-stt/issues/9)) ([6f1134c](https://github.com/MohammadBnei/ukubi-stt/commit/6f1134cc9eaa124a6fe81ee76553c0c9e35e362a)), closes [#8](https://github.com/MohammadBnei/ukubi-stt/issues/8)

## [0.5.1](https://github.com/MohammadBnei/ukubi-stt/compare/0.5.0...0.5.1) (2026-08-31)


### Bug Fixes

* **streaming:** flush the tail, or every utterance loses its ending ([#8](https://github.com/MohammadBnei/ukubi-stt/issues/8)) ([6a5d3d3](https://github.com/MohammadBnei/ukubi-stt/commit/6a5d3d3cdc03a0586e1b01ecd5326c109951a74f))

# [0.5.0](https://github.com/MohammadBnei/ukubi-stt/compare/0.4.0...0.5.0) (2026-08-31)


### Bug Fixes

* **ci:** bump the image tag after the push, not in the PR ([#6](https://github.com/MohammadBnei/ukubi-stt/issues/6)) ([0159983](https://github.com/MohammadBnei/ukubi-stt/commit/015998394f0350d207975ac978334cc60a5d2a8f))


### Features

* realtime streaming — Nemotron, per-session state, same unary RPC ([#7](https://github.com/MohammadBnei/ukubi-stt/issues/7)) ([b8e7a93](https://github.com/MohammadBnei/ukubi-stt/commit/b8e7a93ddc2b732b4ef2c5aa8687a14e857484a4))

# [0.4.0](https://github.com/MohammadBnei/ukubi-stt/compare/0.3.0...0.4.0) (2026-08-31)


### Features

* **web:** browser test client, served from the same origin ([#5](https://github.com/MohammadBnei/ukubi-stt/issues/5)) ([c044a5f](https://github.com/MohammadBnei/ukubi-stt/commit/c044a5f964e34d9a38b7356f032b9431d796cb6e))
