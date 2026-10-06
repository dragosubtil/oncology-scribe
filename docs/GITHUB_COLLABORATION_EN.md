# Team Collaboration

Oncology Scribe is a research prototype for multilingual oncology transcription,
audio-linked review and source-grounded note drafting. Use a private repository
for team development. GitHub hosts the source code; each colleague runs the
Python application locally. GitHub Pages cannot run this backend.

## Get Started

Accept the repository invitation, then clone it using your own GitHub account:

```powershell
git clone https://github.com/dragosubtil/oncology-scribe.git
cd oncology-scribe
```

Follow [the quickstart](QUICKSTART_EN.md) to install locked dependencies, prepare
an eligible model and start the app. The address `127.0.0.1` refers to your own
computer, not another team member's server. Keep the server on loopback.

## Make Changes

1. Create a branch for a focused change.
2. Add regression tests for any bug fix. Do not weaken audio identity checks,
   medication safeguards or human review requirements to improve a score.
3. Run the Python tests, Ruff and the JavaScript tests listed in the README.
4. Restart the server and check the affected workflow in the browser.
5. Review and stage only the intended files, then open a pull request for review.
6. Have another colleague verify installation and the changed workflow before
   merging. Follow the course's submission and contribution requirements.

## Keep Out of Git

Do not commit patient recordings, identifying information, credentials, model
weights, external datasets, environments, generated exports or diagnostic logs.
Only the explicitly catalogued fictional synthetic audio examples are bundled.
The `.gitignore` is a precaution, not proof that a commit is safe.

Check the staged diff and run a credential scan before pushing. If a real secret
has been committed, deleting the file alone does not remove it from Git history.
See [GitHub's sensitive-data guidance](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).

Review [third-party notices](THIRD_PARTY_NOTICES.md) before redistribution.
The prototype is not clinically validated and requires human review.
