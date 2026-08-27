# Using GitHub to the fullest (for this repo)

A tour of GitHub's features, ordered roughly by usefulness for a thesis project.
Your repo: https://github.com/Adraxem/thesis_experiment

## 0. Set your commit identity (do once, if you haven't)
```bash
git config --global user.name  "Ardacan Yildiz"
git config --global user.email "your-github-email@example.com"
```
Use the email tied to your GitHub account so commits link to your profile (this is what
makes your green contribution squares fill in).

## 1. Branches + Pull Requests (the professional workflow)
Instead of committing straight to `main`, do work on a branch and merge via a Pull Request.
```bash
git checkout -b feature/fig6-training-curve   # new branch
# ...edit, commit...
git push -u origin feature/fig6-training-curve
```
Then on GitHub click "Compare & pull request" → open the PR → Merge. PRs give you a diff
view, a place to write what changed, and (bonus) they trigger achievements like
**Pull Shark** and **YOLO**. Delete the branch after merging.

## 2. Tags + Releases (snapshot your milestones)
A **tag** marks a commit; a **Release** is a tag with notes + downloadable assets. Perfect
for thesis checkpoints. Use semantic-ish versions:
```bash
git tag -a v0.1-scaffold -m "Working scaffold: sweep, predictor, optimizer, training capture"
git push origin v0.1-scaffold
```
Then on GitHub: Releases → "Draft a new release" → pick the tag → write notes → attach the
thesis PDF or a results zip → Publish. Good milestones: `v0.1-scaffold`, `v0.2-orin-data`,
`v1.0-defense`.

## 3. License (you have MIT)
`LICENSE` = MIT: anyone can use/modify with attribution, no warranty. Good default for
open research code. To change it, GitHub → Add file → the license picker, or edit LICENSE.
GitHub shows the license in the repo's right sidebar automatically.

## 4. "Cite this repository" (academic — you have CITATION.cff)
`CITATION.cff` makes GitHub show a **Cite this repository** button that spits out APA/BibTeX.
Update the author email/date in it. Great for a thesis repo.

## 5. README badges (the status stickers)
Add these to the TOP of README.md (edit the URLs to your repo):
```markdown
![CI](https://github.com/Adraxem/thesis_experiment/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
```
The first turns green when CI passes (see next).

## 6. GitHub Actions / CI (you now have .github/workflows/ci.yml)
On every push, GitHub spins up a fresh Ubuntu machine, installs deps, byte-compiles the
code, and runs the synthetic-data smoke test. A green check = "nothing is broken." See it
under the **Actions** tab. It's free for public repos and is what powers the CI badge.
Running Actions also unlocks the CI/CD experience employers look for.

## 7. Issues + Milestones + Projects (track the thesis to Nov 13)
- **Issues**: one per task ("Verify nvpmodel IDs on Orin", "Collect real dataset",
  "Build fig6"). Reference them in commits with `#12` to auto-link; write `Closes #12` in
  a PR to auto-close.
- **Milestones**: group issues by deadline (e.g. "Data collection", "Defense draft").
- **Projects** (Projects tab): a kanban board (Todo / Doing / Done) over your issues.
This replaces a scattered TODO list and gives your advisor visibility.

## 8. GitHub Pages (host your figures/dashboard for free)
Settings → Pages → deploy from `main` /docs or /root. Drop an HTML dashboard of your
results there and it's live at `https://adraxem.github.io/thesis_experiment/`. Nice for
sharing plots without sending files.

## 9. Make it pip-installable (optional, "package" it)
Add a `pyproject.toml` and you (or others) can `pip install git+https://...`. Sketch:
```toml
[project]
name = "edge-power-thesis"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["numpy","pandas","scikit-learn","matplotlib","scipy"]
```
For a thesis this is optional — nice-to-have, not required. (GitHub Packages / PyPI
publishing is the next step up, usually overkill here.)

## 10. Housekeeping that makes the repo look sharp
- `.gitignore` — you have one (keeps venvs/caches out).
- `.gitattributes` — add `* text=auto` and `*.bat text eol=crlf` so line endings stay sane
  across Windows/Orin (your .bat needs CRLF).
- **About** box (top-right of the repo page): add a description + topics like
  `jetson`, `edge-ai`, `power-measurement`, `pytorch`, `tensorrt`.
- **Pin** the repo on your profile.
- Enable **Dependabot** (Settings → Security) for dependency alerts.

## 11. Achievements (the "booji" badges on your profile)
GitHub shows achievement badges on your profile. The set evolves, but common ones and how
to earn them:
- **Pull Shark** — get PRs merged (use the branch+PR flow in §1).
- **YOLO** — merge a PR without review (solo repos do this naturally).
- **Quickdraw** — close an issue or PR within 5 minutes of opening.
- **Pair Extraordinaire** — co-authored commits (add `Co-authored-by:` trailers).
- **Galaxy Brain** — accepted answers in GitHub Discussions (enable Discussions).
- **Starstruck** — get your repo starred (16+/32+ stars tiers).
- **Public Sponsor** / **Heart On Your Sleeve** / **Open Sourcerer** — sponsoring,
  reactions, working across many public repos.
See yours at github.com/Adraxem?tab=achievements (the set + rules change over time, so
treat this as a guide, not gospel).

## 12. Your daily loop, unchanged
```bash
git add .
git commit -m "what changed"
git push
```
Everything above is layered ON TOP of that. Start with §1 (branches/PRs) and §2 (releases);
the rest you add as the thesis grows.
