# TenfloweRS Release Checklist

## Pre-flight
- [ ] Branch name matches target version (e.g. `0.1.1` for v0.1.1 release)
- [ ] `Cargo.toml` workspace.package.version is updated to the target version
- [ ] All planned `TODO.md` items for this release are marked `[x]`
- [ ] `CHANGELOG.md` has an entry for the target version with date

## Validate
- [ ] `cargo nextest run --all-features` — all tests pass, 0 failures
- [ ] `cargo clippy --all-features --all-targets -- -D warnings` — 0 warnings
- [ ] `cargo doc --all-features --no-deps` — no missing-doc warnings
- [ ] `rslines 50` confirms all files < 2000 lines
- [ ] No `unwrap()` in production code (`grep -r "\.unwrap()" --include="*.rs" src/ | grep -v "test\|bench\|example"`)

## Version bump
- [ ] Run `scripts/bump_version.sh <new-version>` and verify the diff
- [ ] Commit version bump with message `chore: bump version to X.Y.Z`
- [ ] Create git tag: `git tag -s vX.Y.Z -m "Release vX.Y.Z"`

## Publish (dry-run first)
- [ ] Run `./scripts/publish_meta.sh` (dry-run) — all crates validate
- [ ] Set `TENFLOWERS_PUBLISH_CONFIRM=1 ./scripts/publish_meta.sh` — publish

## Post-release
- [ ] Push tag: `git push origin vX.Y.Z`
- [ ] Create GitHub release with CHANGELOG notes
- [ ] Announce in project channels
