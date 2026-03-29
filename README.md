# REPLIT-QUANTUM-MCAGI-NO-LLM
AI built on quantum physics — no LLM

## Creating a Release

Instead of uploading a zip file manually from a device (which can fail with
`tls: bad record MAC` errors on mobile/Termux connections), use a Git tag to
trigger an automated GitHub Actions release:

```bash
# 1. Make sure your latest code is committed and pushed
git add .
git commit -m "Your commit message"
git push origin main

# 2. Create and push a version tag — this starts the release workflow
git tag v1.0
git push origin v1.0
```

GitHub Actions will then:
1. Check out the repository on GitHub's own servers
2. Package everything into `quantum_mcagi_v1.0.zip` (excluding `.git`, `venv`, bytecode, etc.)
3. Publish a new GitHub Release with the zip attached automatically

No manual file upload needed — the packaging and upload happen server-side,
so flaky mobile TLS connections are never a problem.
