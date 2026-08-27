# Git cheatsheet (for this thesis)

## Mental model
Git takes **snapshots** ("commits") of your folder. Each commit is a save point you
can return to, compare against, or undo. A "repo" is the folder + its full history
(stored in a hidden `.git` subfolder). **GitHub** is a website that hosts a copy of
your repo online so you can pull it onto other machines (like the Orin) and back it up.

## One-time setup (per machine)
    git --version                              # check it's installed
    git config --global user.name  "Ardacan Yildiz"
    git config --global user.email "cgamerstr@gmail.com"

## Start tracking THIS folder (do once, on your PC)
    cd "%USERPROFILE%\OneDrive\Desktop\thesis_experiment"
    git init                 # create the repo (.git folder)
    git add .                # stage everything (respecting .gitignore)
    git commit -m "Initial commit: edge-inference thesis scaffold"

## The daily loop (repeat forever)
    git status               # what changed?
    git add <file>           # stage a specific file...
    git add .                # ...or everything
    git commit -m "message"  # save a snapshot of the staged changes
    git log --oneline        # list your snapshots

## Put it on GitHub (so you can clone on the Orin)
1. Make an empty repo on github.com (no README), copy its URL.
2. Link and push:
    git remote add origin https://github.com/<you>/thesis_experiment.git
    git branch -M main
    git push -u origin main          # first push; asks for a Personal Access Token
   (GitHub wants a token, not your password — make one at github.com >
    Settings > Developer settings > Personal access tokens > Fine-grained,
    give it repo access, paste it as the password.)

## After the first push, publishing new work is just:
    git add .
    git commit -m "what I changed"
    git push

## Clone onto the Orin (or any machine)
    git clone https://github.com/<you>/thesis_experiment.git
    cd thesis_experiment

## The PC <-> Orin round trip
- PC: make changes -> add/commit/push
- Orin: `git pull`  (pulls your latest code)
- Orin: run the sweep, which writes data/results -> add/commit/push
- PC:  `git pull`  (gets the real measurements back)

## Undo / safety
    git diff                 # see unstaged changes
    git checkout -- <file>   # discard changes to a file (careful!)
    git revert <hash>        # make a NEW commit that undoes an old one (safe)
    git log --oneline        # find a commit hash

## WARNING about OneDrive
This folder is inside OneDrive. OneDrive and Git can occasionally fight over the
`.git` folder while syncing. It usually works, but if you see weird corruption,
move the repo OUT of OneDrive (e.g. C:\Users\ardac\code\thesis_experiment) and let
GitHub be your backup instead of OneDrive.
