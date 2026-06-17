# CHANGELOG

<!-- version list -->

## v1.1.0 (2026-06-16)

### Features

- **recognizers**: Add InstitutionRecognizer for institution-specific PHI detection
  ([#7](https://github.com/susom/tide2-core/pull/7),
  [`6f6ae38`](https://github.com/susom/tide2-core/commit/6f6ae38b10e13c1f79cdd778656506acfb9cff1f))

- Require ML stack, upgrade to transformers 5, and add PyPI publishing
  ([#17](https://github.com/susom/tide2-core/pull/17),
  [`13e3047`](https://github.com/susom/tide2-core/commit/13e3047636b97c7e30e104c930a97219ce6acba4))

- **notebooks**: Make tide2_pipeline runnable in Google Colab
  ([#20](https://github.com/susom/tide2-core/pull/20),
  [`336c194`](https://github.com/susom/tide2-core/commit/336c194b3726ada65803c3bb4844eb5bcd09b31e))

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
