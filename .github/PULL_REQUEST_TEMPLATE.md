## Summary

<!-- Describe what this PR does and why. Two to three sentences. -->

## Changes

<!--
List the concrete changes introduced. Be specific.
Example:
- Added retry logic to `ToolExecutor` with configurable backoff
- Updated `Agent.run()` signature to accept timeout parameter
-->

-

## Related

<!-- Issues closed by this PR. Example: Closes #12, Closes #34 -->

Closes #

## Commit type

- [ ] `feat` — new feature
- [ ] `fix` — bug fix
- [ ] `docs` — documentation only
- [ ] `refactor` — code restructuring, no behaviour change
- [ ] `test` — adding or correcting tests
- [ ] `chore` — maintenance, deps, config, tooling
- [ ] `perf` — performance improvement
- [ ] `ci` — CI/CD pipeline changes

## Release

<!-- Apply exactly one bump label to this PR if it should trigger a release. Leave unchecked if no release is needed. -->

- [ ] `bump:patch` applied — backwards-compatible bug fix
- [ ] `bump:minor` applied — new backwards-compatible functionality
- [ ] `bump:major` applied — breaking change (`!` in commit message)
- [ ] No release needed

## Review checklist

- [ ] Code is clear and follows project conventions
- [ ] No secrets or credentials introduced
- [ ] New behaviour is covered by tests; existing tests pass
- [ ] Public API changes are reflected in docstrings and docs
- [ ] CI is green (`lint`, `type-check`, `run-pytest`)
- [ ] Correct bump label applied (or confirmed not needed)
