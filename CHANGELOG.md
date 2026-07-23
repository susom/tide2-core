# CHANGELOG

<!-- version list -->

## [1.2.3](https://github.com/susom/tide2-core/compare/v1.2.2...v1.2.3) (2026-07-23)


### Bug Fixes

* **anonymizer:** pin presidio 2.2.364 and harden patches against merge-method rename ([#39](https://github.com/susom/tide2-core/issues/39)) ([a3c1812](https://github.com/susom/tide2-core/commit/a3c181280bcd581b266fa0e80f31de2f041764f3))

## [1.2.2](https://github.com/susom/tide2-core/compare/v1.2.1...v1.2.2) (2026-07-22)


### Bug Fixes

* **recognizers:** use identical recognizer_name for live and reused transformer results ([#33](https://github.com/susom/tide2-core/issues/33)) ([f5da675](https://github.com/susom/tide2-core/commit/f5da675d9e18d60bad2382cdd7b95bc833616408))
* **transformers:** fail fast on missing absolute model_path and reconcile local_files_only ([#30](https://github.com/susom/tide2-core/issues/30)) ([281ff54](https://github.com/susom/tide2-core/commit/281ff543b094c182fb2a5682e65f23969c2c4418))


### Documentation

* add CLAUDE.md for agent onboarding ([#29](https://github.com/susom/tide2-core/issues/29)) ([bb360db](https://github.com/susom/tide2-core/commit/bb360db49b583fba5a97a1d002685e6bba6659f5))


### Build System

* **deps:** add upper-bound version constraints to pyproject.toml ([#34](https://github.com/susom/tide2-core/issues/34)) ([353ceed](https://github.com/susom/tide2-core/commit/353ceed46fc257543ff819428a37dbd5d46b32ef))

## [1.2.1](https://github.com/susom/tide2-core/compare/v1.2.0...v1.2.1) (2026-07-21)


### Bug Fixes

* **recognizer:** stop eager en_core_web_lg load crashing actors ([#28](https://github.com/susom/tide2-core/issues/28)) ([67d10e8](https://github.com/susom/tide2-core/commit/67d10e8c7e13ede6fb881a9e5e20bac68a80dbb6))

## [1.2.0](https://github.com/susom/tide2-core/compare/v1.1.1...v1.2.0) (2026-06-18)


### Features

* **runner:** add fractional-CPU and checkpoint knobs for small boxes ([e5b46ab](https://github.com/susom/tide2-core/commit/e5b46abe08beba86c3f697cc5ed0252d39101f95))


### Bug Fixes

* **cli:** forward worker_num_cpus and enable_checkpoint to llm-recognizer ([6b2b549](https://github.com/susom/tide2-core/commit/6b2b54972561cd0980972f216536cb8a0703b5e7))

## v1.0.0 (2026-06-12)

_This release is published under the MIT License._

### Bug Fixes

- Fetch base SHA before pre-commit diff on PRs
  ([#14](https://github.com/susom/tide2-core/pull/14),
  [`3a4e4d2`](https://github.com/susom/tide2-core/commit/3a4e4d23da60ae120beb5aa40f44ab9b8738455d))

- **anonymizer**: Resolve patient_uid name collision causing silent 0-row output
  ([`34e0619`](https://github.com/susom/tide2-core/commit/34e0619c1644f8fbf7b895482d993bd2dbf32c59))

### Documentation

- Correct README and runner docstrings to match code
  ([`f4b09b3`](https://github.com/susom/tide2-core/commit/f4b09b35655d80cc68d6cbe5b878b0e47ede2644))

- Remove pre-release private/do-not-publish notice
  ([`fabef8f`](https://github.com/susom/tide2-core/commit/fabef8f4d6cbcb405b7351fb8e20b5ba752416e1))

- Update clone URL and venv path to tide2-core
  ([`d440c40`](https://github.com/susom/tide2-core/commit/d440c40719afafb93f821e70739de574c651a677))

### Features

- Add OSSF Scorecard support (score 9.1/10)
  ([#11](https://github.com/susom/tide2-core/pull/11),
  [`6ff5249`](https://github.com/susom/tide2-core/commit/6ff5249dc8c0f7bcef803089d36f70a12d6bae87))

- Add tide2-core PHI de-identification pipeline
  ([`f165b26`](https://github.com/susom/tide2-core/commit/f165b26bbd849c2eb4a13c3cd5ffd884cad15530))

### Performance Improvements

- **pipeline**: Auto-tune CPU resource allocation for Mac/CPU runs
  ([`c96ec32`](https://github.com/susom/tide2-core/commit/c96ec32b1ee614c5d90550743355a604eed3260d))
