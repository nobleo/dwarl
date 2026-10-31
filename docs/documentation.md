# Maintaining the documentation

The site is built from `docs/` with MkDocs. Search and navigation expose the
workflow; command help and resolved configuration are generated directly from
the implementation at build time. Update the walkthrough when behavior changes.

Preview without installing the training stack:

```bash
python3 -m venv /tmp/dwarl-docs
/tmp/dwarl-docs/bin/pip install -r docs/requirements.txt
/tmp/dwarl-docs/bin/mkdocs serve
```

CI builds with `mkdocs build --strict` on every main push and pull request and
uploads a downloadable `documentation` artifact. This catches broken internal
links and keeps the generated reference in sync with code.

## GitHub Pages

The current organization plan does not support Pages for this private repository.
The repository stays private. Once its plan supports Pages (or the owner chooses
to publish the repository), enable **Settings → Pages → GitHub Actions** and set
the repository Actions variable `PAGES_ENABLED` to `true`. Re-run Documentation
or push to main. Subsequent main pushes automatically deploy the site at
<https://nobleo.github.io/dwarl/>.

Pages site visibility is separate from repository visibility; check the intended
site access when enabling hosting. Until then, use the CI artifact or local preview.
