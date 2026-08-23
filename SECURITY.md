# Security policy

## Supported version

Security fixes are applied to the latest revision of the `main` branch.

## Reporting a vulnerability

Please do not disclose a suspected vulnerability in a public issue. Use
GitHub's private vulnerability reporting feature on the repository Security
page. Include reproduction steps, affected files, and potential impact.

Research checkpoints are loaded through PyTorch/Stable-Baselines3. Only load
checkpoint files obtained from the official GitHub Releases page and verify
the published SHA-256 digest before use.
