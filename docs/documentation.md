# Maintaining the documentation

The site is built from `docs/` with MkDocs. Search and navigation expose the
workflow; command help and resolved configuration are generated directly from
the implementation at build time. Update the walkthrough when behavior changes.

Preview without installing the training stack (a slim pixi env with only
`docs/requirements.txt`'s packages):

```bash
pixi run docs-serve      # mkdocs serve
pixi run docs            # mkdocs build --strict, as in CI
```

CI builds with `mkdocs build --strict` on every main push and pull request and
uploads a downloadable `documentation` artifact. This catches broken internal
links and keeps the generated reference in sync with code.

## GitHub Pages

To host the site, enable **Settings → Pages → GitHub Actions** and set the repository
Actions variable `PAGES_ENABLED` to `true`. Re-run Documentation or push to main.
Subsequent main pushes automatically deploy the site at <https://nobleo.github.io/dwarl/>.
Until then, use the CI artifact or local preview.
