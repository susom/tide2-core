# CHANGELOG

<!-- version list -->

## [1.5.0](https://github.com/susom/tide2-core/compare/v1.4.0...v1.5.0) (2026-11-18)


### Features

* **anonymizer:** add anonymization policy and apply it to notes and cells ([#119](https://github.com/susom/tide2-core/issues/119)) ([24e2bba](https://github.com/susom/tide2-core/commit/24e2bba5d568023be00ac48a49a4b072396b3ec0))
* **anonymizer:** add keep_entities to leave chosen entity types unanonymized ([#116](https://github.com/susom/tide2-core/issues/116)) ([2cb9f44](https://github.com/susom/tide2-core/commit/2cb9f445e30faa0ba0d5ebb7973fbe187fe58cf4))
* **anonymizer:** add keyed hash method for id entity types ([#120](https://github.com/susom/tide2-core/issues/120)) ([79f90a0](https://github.com/susom/tide2-core/commit/79f90a03f32910b05b5b7b673371b89089c38bad))
* **anonymizer:** add score thresholds for transformer and other recognizers ([#123](https://github.com/susom/tide2-core/issues/123)) ([a2b6992](https://github.com/susom/tide2-core/commit/a2b6992b521d493554c6f082a280c86889a51f05))
* **anonymizer:** add strict and relaxed zip code anonymizer ([#113](https://github.com/susom/tide2-core/issues/113)) ([ceb0ad5](https://github.com/susom/tide2-core/commit/ceb0ad566cb71488676d3e2d40dbdc1da186aea6))
* **anonymizer:** anonymize free-text cells with precomputed spans ([#114](https://github.com/susom/tide2-core/issues/114)) ([b55f40b](https://github.com/susom/tide2-core/commit/b55f40b87864a059bceaeec0e1df10c30fd46922))
* **anonymizer:** cap shifted dates at the 90th birthday ([#111](https://github.com/susom/tide2-core/issues/111)) ([4b27470](https://github.com/susom/tide2-core/commit/4b2747077b1964e879c0c16130d4b46ca4759501))
* **anonymizer:** fix date jitter parsing and add a configurable jitter range ([#107](https://github.com/susom/tide2-core/issues/107)) ([22ff87a](https://github.com/susom/tide2-core/commit/22ff87a45634cc5c9a19fc03f82eb7e1456a08b7))
* **anonymizer:** make accession number hashing opt-in ([#115](https://github.com/susom/tide2-core/issues/115)) ([ff203d3](https://github.com/susom/tide2-core/commit/ff203d3cb7b677456e1a190597fcd1dde938e73d))
* **anonymizers:** add cell-level anonymization function ([#106](https://github.com/susom/tide2-core/issues/106)) ([2df9bc7](https://github.com/susom/tide2-core/commit/2df9bc7de12bb14a11a301b0c315105e62f2a36c))
* **anonymizer:** shift day, month and year parts with their full date ([#112](https://github.com/susom/tide2-core/issues/112)) ([bd2e99e](https://github.com/susom/tide2-core/commit/bd2e99e4559c1c42a81b2fa35ba8e0d6d348282c))
* **cryptographic:** add relaxed fpe mode for int64 cells ([#117](https://github.com/susom/tide2-core/issues/117)) ([a608131](https://github.com/susom/tide2-core/commit/a608131b3a60d3c77499e02a2a916a8545794d4e))
* **runner:** default standalone stage checkpointing to row_id and expose --id-column ([#101](https://github.com/susom/tide2-core/issues/101)) ([370d954](https://github.com/susom/tide2-core/commit/370d954bef552bcf5907bd6fe4d5d313e58317cd)), closes [#98](https://github.com/susom/tide2-core/issues/98)
* **visualizer:** simplify visualizer with parquet ingestion and streamlined ui ([#102](https://github.com/susom/tide2-core/issues/102)) ([58508cb](https://github.com/susom/tide2-core/commit/58508cb36331d4c01358ab5e60a39a31c6592f7a))


### Bug Fixes

* **anonymizers:** preserve casing in hospital and institution replacements ([#95](https://github.com/susom/tide2-core/issues/95)) ([5630680](https://github.com/susom/tide2-core/commit/563068059206cf239d95f2b75561f136aca6b9b0))
* **cryptographic:** replace FF3 with collision-free FPE on FF1 ([#108](https://github.com/susom/tide2-core/issues/108)) ([2fb52a8](https://github.com/susom/tide2-core/commit/2fb52a8b55baee0f20348a21008c6a938a2414c2))


### Documentation

* **notebooks:** use write_visualizer_json in pipeline notebook ([#93](https://github.com/susom/tide2-core/issues/93)) ([c023d93](https://github.com/susom/tide2-core/commit/c023d93dad2edd106ba21b9791910bcf01e9b86a))
* update documentation, add distributed guide, and resolve doc gaps ([#97](https://github.com/susom/tide2-core/issues/97)) ([68aae8a](https://github.com/susom/tide2-core/commit/68aae8ad8cb7bfb11968f3f3767ab8fd96839eeb))

## [1.4.0](https://github.com/susom/tide2-core/compare/v1.3.1...v1.4.0) (2026-10-13)


### Features

* **runner:** rework the ray pipeline for token-accurate ner, direct actors, and hardware autotuning ([#62](https://github.com/susom/tide2-core/issues/62)) ([f8cd65f](https://github.com/susom/tide2-core/commit/f8cd65f5dfaf70ba6c5810da1ef864607dee3ffb))

## [1.3.1](https://github.com/susom/tide2-core/compare/v1.3.0...v1.3.1) (2026-09-23)


### Bug Fixes

* **anonymizer:** scope accession hash to per-note patient_uid ([#59](https://github.com/susom/tide2-core/issues/59)) ([5939034](https://github.com/susom/tide2-core/commit/593903430daafcb6db9b8c1ced1c316f6f269967))
* **deps:** bump vulnerable dependencies to patched versions ([#57](https://github.com/susom/tide2-core/issues/57)) ([b7ee611](https://github.com/susom/tide2-core/commit/b7ee61169bd612683221c8ee7b0a608651546eca))

## [1.3.0](https://github.com/susom/tide2-core/compare/v1.2.4...v1.3.0) (2026-08-26)


### Features

* **transformers:** add sentry-clinical-ner model config and default the pipeline notebook to it ([#55](https://github.com/susom/tide2-core/issues/55)) ([d5e0231](https://github.com/susom/tide2-core/commit/d5e023195c3de032f8be5ebcc54b9af303d960cf))

## [1.2.4](https://github.com/susom/tide2-core/compare/v1.2.3...v1.2.4) (2026-07-24)


### Documentation

* **bdf:** align bdf.yaml with actual repo capabilities ([#37](https://github.com/susom/tide2-core/issues/37)) ([4702772](https://github.com/susom/tide2-core/commit/47027722189accdfdca976e0296051ede22e0b6c))
* **notebook:** clean torch stack and bump tide2 to 1.2.3 for Colab ([#38](https://github.com/susom/tide2-core/issues/38)) ([5b81e92](https://github.com/susom/tide2-core/commit/5b81e92007dce0cc0a284a86b55f47ec0d62babc))

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
