# Replit ↔ GitHub Migration Guide

This guide explains how to connect your Replit project to this GitHub
repository so that code flows in both directions.

---

## Option A — Import this GitHub repo into Replit (recommended)

1. Open **[replit.com](https://replit.com)** and click **Create Repl**.
2. Choose **Import from GitHub**.
3. Paste the repository URL:
   ```
   https://github.com/Bonsib420/REPLIT-QUANTUM-MCAGI-NO-LLM
   ```
4. Replit will detect the `.replit` and `replit.nix` files and configure
   the environment automatically.
5. Click **Run** to start `main.py`.

Any changes you make in Replit can be pushed back to GitHub using the
built-in Git panel (see *"Pushing changes"* below).

---

## Option B — Connect an existing Replit project to this repo

If you already have a Replit project with code and want to push it here:

### 1. Open the Git panel in Replit

In your Replit project, open the **Version Control** tab (Git icon in the
left sidebar).

### 2. Link the remote

If no remote is set, open the **Shell** tab and run:

```bash
git remote add origin https://github.com/Bonsib420/REPLIT-QUANTUM-MCAGI-NO-LLM.git
```

### 3. Pull the scaffolding (merge with your code)

```bash
git fetch origin main
git merge origin/main --allow-unrelated-histories
```

Resolve any conflicts, then commit.

### 4. Push your code

```bash
git push -u origin main
```

You may need to authenticate — Replit can use a GitHub token.  Go to
**Replit Settings → Connected Services → GitHub** to link your account.

---

## Pushing changes from Replit to GitHub

1. Open the **Version Control** panel.
2. Stage your changes, write a commit message, and click **Commit & Push**.

Or from the Shell:

```bash
git add .
git commit -m "describe your changes"
git push
```

---

## Pulling GitHub changes into Replit

From the Shell:

```bash
git pull origin main
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Permission denied` on push | Link your GitHub account in Replit Settings → Connected Services |
| Merge conflicts | Open conflicting files, choose which changes to keep, then `git add .` and `git commit` |
| `.replit` missing after clone | This repo already includes `.replit` and `replit.nix` — they should be present |
| Wrong Python version | Edit `replit.nix` to change `pkgs.python311` to your preferred version |
