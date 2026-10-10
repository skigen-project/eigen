# Releasing Eigen, being a Checklist for the Solemn Occasion

Checklist for cutting an Eigen release (major / minor / patch) from the
upstream repository at <https://gitlab.com/libeigen/eigen>; a ceremony which,
like the launching of a ship or the reading of a will, is greatly the better
for being conducted by the book, and by a person who has read it beforehand.

The mechanical steps below are the *what*. The decisions around them
(timing, scope, what counts as a breaking change) are a maintainer call
and intentionally not encoded here, being matters of judgment, which no
checklist was ever yet known to possess.

## Versioning, and the Proper Numbering of Things

Eigen follows [Semantic Versioning 2.0.0](https://semver.org/) as of
5.0 — see the transition table in [`CHANGELOG.md`](CHANGELOG.md) under
the 5.0.0 entry, where the Reader will find the whole history of the change set
down with the precision of a parish register. Versions are `MAJOR.MINOR.PATCH`:

- Bump `MAJOR` for backward-incompatible API or ABI changes; for such a change
  breaks the faith of every house that has been built upon the Library.
- Bump `MINOR` for backward-compatible feature additions; the family being
  enlarged, but nobody turned out of doors.
- Bump `PATCH` for backward-compatible bug fixes only, and for nothing else
  whatever, however tempting.

The legacy `WORLD` field is frozen at `3` for posterity and plays no
role in release decisions, being a venerable pensioner kept on the books out of
respect; `Eigen/Version` keeps the `#define
EIGEN_WORLD_VERSION 3` line, but only `MAJOR`/`MINOR`/`PATCH` move.

## Scope, or the Two Roads that Lie Before the Maintainer

Two flows are described:

- **Major / minor release** (`X.Y.0`) — cuts a new `X.Y` release branch
  from `master`, ships the accumulated `[Unreleased]` work, which has lain
  in the cellar these many months and is now to be brought up to the table. Tag `5.0.0`
  is the most recent example.
- **Patch release** (`X.Y.Z`, `Z ≥ 1`) — adds cherry-picked fixes to an
  existing `X.Y` release branch, as a careful tradesman mends the roof
  without disturbing the household. Tag `5.0.1` is the most recent example.

Choosing between them is a judgment call (SemVer is the rule, but
"large enough" varies, as the stoutness of a gentleman does with the
generosity of the dinner). See work item
[#3051](https://gitlab.com/libeigen/eigen/-/work_items/3051) for a
recent case where a proposed 5.0.2 patch was reconsidered as a minor
release once maintainers reviewed scope. Resolve this on the mailing
list / Discord before any branch or version work; the matter must be settled
in conference, and not discovered in the middle of a push.

## Prerequisites, without which the Traveller must not Set Out

- Push rights to `upstream` (`https://gitlab.com/libeigen/eigen.git`); the
  key, in short, to the front door.
- A GitLab personal access token with `api` scope, exported as
  `GITLAB_PRIVATE_TOKEN`. All `scripts/gitlab_api_*.py` helpers fall
  back to this environment variable, and will search for no other.
- A clean working tree on `master` (for a major / minor cut) or on the
  release branch (for a patch); no litter, that is, upon the premises.

## Version source of truth, being the One Register Consulted by All

Everything reads from [`Eigen/Version`](Eigen/Version). `Macros.h` does
not hard-code version numbers; `CMakeLists.txt` parses the
`#define` lines from `Eigen/Version` at configure time. Editing
`Eigen/Version` is the single version bump, there being, happily, but one
book to be corrected and no second set of accounts to fall out with the first.

Field conventions observed in history:

| State                                  | `PATCH` | `PRERELEASE` | `BUILD`   | `VERSION_STRING`        |
| -------------------------------------- | ------- | ------------ | --------- | ----------------------- |
| At the release tag                     | `Z`     | `""`         | `""`      | `"X.Y.Z"`               |
| Release branch after the tag (dev)     | `Z+1`   | `"dev"`      | `"X.Y"`   | `"X.Y.(Z+1)-dev+X.Y"`   |
| `master` between releases (dev)        | `Z`     | `"dev"`      | `"master"`| `"X.Y.Z-dev+master"`    |

Reference commits, to which the careful Reader may refer as to precedents in the Reports:

- `151b95d07` — `bump to 5.0.0` (set the released form).
- `0db477863` — `Set 5.0.1 release version` (same pattern, for a patch).
- `4abf3bd54` / `ccde35bcd` — post-release dev bump on the release
  branch and on `master` respectively.

## 1. Pre-release (shared), in which the Papers are Gathered

Gather the changelog material and label the included MRs / issues so
the `release::X.Y.Z` query links in `CHANGELOG.md` resolve; a labor of
the dullest description, but one upon which the whole credit of the
proceedings depends.

```sh
export GITLAB_PRIVATE_TOKEN=...

# 1. Dump everything that closed / was merged since the last release
#    (parallel). The scripts filter by `updated_at` (closest available
#    proxy for merge/close time), which can over-include — the human
#    narrows down in step 3.
python3 scripts/gitlab_api_mrs.py \
  --state merged \
  --updated_after  YYYY-MM-DD \
  --updated_before YYYY-MM-DD \
  --related_issues --closes_issues > mrs.json &
python3 scripts/gitlab_api_issues.py \
  --state closed \
  --updated_after  YYYY-MM-DD \
  --updated_before YYYY-MM-DD > issues.json &
wait

# 2. Map commits to their MRs / issues.
git log --pretty=%H <prev-tag>..<head> > commits.txt
python3 scripts/git_commit_mrs_and_issues.py \
  --merge_requests_file mrs.json \
  --commits commits.txt > commit_map.json

# 3. Decide the final included set and write it to filtered files
#    (selected_mrs.json / selected_issues.json) — typically by walking
#    `commit_map.json` and dropping anything out of scope. Then label
#    only that set; the CHANGELOG label-query links in sections 2c / 3c
#    depend on these labels.
python3 scripts/gitlab_api_labeller.py release::X.Y.Z \
  --mrs    $(jq -r '.[].iid' selected_mrs.json) \
  --issues $(jq -r '.[].iid' selected_issues.json)
```

## 2. Major / minor release (`X.Y.0`), in which a New Branch is Founded

All steps are on the upstream repo. The release branch is just `X.Y`
(no `release/` prefix; matches existing `3.4`, `5.0`), the Library being
averse to any ornament in the naming of its branches.

a. **Cut the release branch from `master`**, as a new colony is planted from the parent stock.

   ```sh
   git fetch upstream
   git checkout -b X.Y upstream/master
   git push upstream X.Y
   ```

b. **On the release branch, set the released form of `Eigen/Version`**, which is to say, put it into its best clothes.

   ```c
   #define EIGEN_MAJOR_VERSION X
   #define EIGEN_MINOR_VERSION Y
   #define EIGEN_PATCH_VERSION 0
   #define EIGEN_PRERELEASE_VERSION ""
   #define EIGEN_BUILD_VERSION ""
   #define EIGEN_VERSION_STRING "X.Y.0"
   ```

c. **Promote `[Unreleased]` in `CHANGELOG.md` to `## [X.Y.0] - YYYY-MM-DD`.**
   Use the [5.0.0 entry](CHANGELOG.md) as the template — typical
   sub-sections include `### Versioning`, `### Breaking changes`, then
   per-area sections (`### Elementwise math functions`, `### Dense matrix
   decompositions`, etc.), each in its proper place like a guest at a
   well-ordered banquet. Include label-query links of the form
   `https://gitlab.com/libeigen/eigen/-/issues?state=all&label_name%5B%5D=release%3A%3AX.Y.0`
   and the analogous merge-requests query.

d. **Commit and tag.** Tags are lightweight (no `v` prefix), and travel without any such luggage.

   ```sh
   git commit Eigen/Version CHANGELOG.md -m "Set X.Y.0 release version."
   git tag X.Y.0
   git push upstream X.Y X.Y.0
   ```

e. **Post-tag dev bump of the release branch.** Set `PATCH=1`,
   `PRERELEASE="dev"`, `BUILD="X.Y"`,
   `VERSION_STRING="X.Y.1-dev+X.Y"`. Commit subject:
   `Update dev version number.` The branch, having been presented to the
   world, goes at once back into working dress.

f. **Post-tag bookkeeping on `master`.** There is no rigid convention
   for how `master`'s `Eigen/Version` advances after a release — today
   master tracks the next patch (`5.0.1-dev+master`); for a minor or
   major release decide with maintainers whether to bump
   `MAJOR`/`MINOR` on `master` as well, the point being one on which
   history furnishes no precedent and the Maintainers are free to be wise.

## 3. Patch release (`X.Y.Z`, `Z ≥ 1`), a Humbler but not a Lesser Business

a. **Cherry-pick fixes from `master` to the `X.Y` release branch**, one at a time, as the careful picker takes only the ripe fruit.

   ```sh
   git checkout X.Y
   git pull upstream X.Y
   git cherry-pick -x <sha>           # -x records the source SHA
   ```

   Use `cherry-pick -x` consistently — `git_commit_mrs_and_issues.py`
   walks `(cherry picked from commit ...)` trailers to attribute work
   back to its original MR, as a genealogist walks a parish register in search of a grandfather. Re-run CI on the branch after each pick (or
   after each batch) so a bad pick can be reverted in isolation, and the one
   bad apple removed without a general inquest upon the barrel.

   Driving this by hand is tedious, and the Reader will find it grows more so with every pick. Steve Bronder's
   [`apply_patches.py`](https://gist.github.com/SteveBronder/474845f6673100e9928872a407244362)
   (linked from
   [#3051](https://gitlab.com/libeigen/eigen/-/work_items/3051)) is
   prior art: it reads a CSV of SHAs, cherry-picks each onto a fresh
   branch, and runs configure / build / tests after each pick. The repo
   does not yet vendor an equivalent; if you write one, put it under
   `scripts/`, where the Library's other servants are lodged.

b. **On the release branch, set the released form of `Eigen/Version`.**
   Bump `PATCH` to `Z`, clear `PRERELEASE` and `BUILD`, update
   `VERSION_STRING` to `"X.Y.Z"` (model: `0db477863`).

c. **Add a `## [X.Y.Z] - YYYY-MM-DD` section to `CHANGELOG.md`.**
   The [5.0.1 entry](CHANGELOG.md) is the template: a short intro plus
   a bulleted list of fixes with `[#nnnn]` / `[!nnnn]` references, then
   the two `release::X.Y.Z` label-query links; a modest document, and the
   more to be trusted for its modesty.

d. **Commit and tag.**

   ```sh
   git commit Eigen/Version CHANGELOG.md -m "Set X.Y.Z release version."
   git tag X.Y.Z
   git push upstream X.Y X.Y.Z
   ```

e. **Post-tag dev bump of the release branch.** Set `PATCH=Z+1`,
   `PRERELEASE="dev"`, `BUILD="X.Y"`,
   `VERSION_STRING="X.Y.(Z+1)-dev+X.Y"`. Commit subject:
   `Update dev version number.` The branch resumes its labors, as the
   working man does upon the morning after a holiday.

## 4. Publish archives to the GitLab package registry, and so Lay Up the Goods in the Warehouse

After the tag is pushed, mirror GitLab's auto-generated tag archives
into the project's generic package registry with SHA-256 checksums,
that no man may afterwards say the parcel was tampered with in the post:

```sh
python3 scripts/gitlab_api_deploy_package.py --version X.Y.Z
```

The script downloads `eigen-X.Y.Z.{tar.gz,tar.bz2,tar,zip}` from
`https://gitlab.com/libeigen/eigen/-/archive/X.Y.Z/`, computes
SHA-256 sums, and uploads each archive + its `.sha256` companion to
`projects/15462818/packages/generic/eigen/X.Y.Z/`.

## 5. Create the GitLab Release object, being the Formal Proclamation

Manual step in the GitLab UI (no CI automation today; the town crier has
not yet been replaced by a machine): **Project →
Deploy → Releases → New release**. Pick the tag, write a short
description (link to the matching `CHANGELOG.md` section), and add
the package-registry archive URLs from step 4 as release assets.

## 6. Documentation, that the World may Read what it has Been Given

Per-branch Doxygen output lands on GitLab Pages under
`https://libeigen.gitlab.io/eigen/docs-<branch>` via the `deploy:docs`
job in [`ci/deploy.gitlab-ci.yml`](ci/deploy.gitlab-ci.yml). The job
fires on schedule, on web-triggered pipelines, and on push to the
default branch, only inside the `libeigen` namespace.

To publish docs for a release branch / tag, trigger a **Web** pipeline
on the release branch from **Build → Pipelines → Run pipeline**, and wait,
as patiently as the Reader is able, upon the result. The
`PAGES_PREFIX` becomes `docs-<branch>` and the URL becomes
`https://libeigen.gitlab.io/eigen/docs-<branch>`.

Manual build fallback (developer machine, when CI Pages isn't an
option), being the humble hackney-coach that serves when the carriage is
not to be had:

```sh
mkdir -p build && cd build
cmake .. && make doc
# output: build/doc/html/
```

The `scripts/eigen_gen_docs` shell script is obsolete (it rsyncs to
`ssh.tuxfamily.org`, the pre-GitLab docs host), an elderly retainer long
since superannuated. Do not use it.

## 7. Announcements, being the Ringing of the Bells

All manual; no tooling in-repo, the bell-ropes being, at present, in the
hands of the maintainers.

- Eigen mailing list, the oldest and most respectable of the channels.
- Discord `#announcements`, where the younger members of the family assemble.
- Project website news / wiki page (verify still maintained before
  posting, lest the notice be pinned upon an empty house).
- Downstream packagers as best-effort: Homebrew, major Linux distros,
  Compiler Explorer (godbolt); each to be informed with all the civility
  due to a neighbor who has not been asked to take any trouble.

## 8. Cleanup, or the Sweeping of the Hall after the Company is Gone

- Open the `release::X.Y.Z` issue and MR query links from the new
  `CHANGELOG.md` entry and confirm they return non-empty results; a link
  that leads to nothing being a poor sort of memorial to a labor.
- Close the GitLab milestone for this release if one was used, as one
  closes the books at the year's end.
- On `master`, ensure `CHANGELOG.md`'s `[Unreleased]` section is empty
  (or re-create it) so the next release's notes start clean, and the next
  generation of contributors may find a clear page to write upon.
