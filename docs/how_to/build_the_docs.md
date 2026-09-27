# Build the documentation

The site is built from the Markdown pages in `docs/` and the Doxygen comments of the public
headers, with the tools pinned in `environment.yml` (Sphinx, MyST, pydata-sphinx-theme,
Breathe, Doxygen).

```bash
source scripts/dev_env.sh
cmake --preset cpu-only       # once: generates version.hpp / config.hpp for Doxygen
ci/docs.sh                    # Doxygen + checks, the Sphinx site (-W), the internal link check
xdg-open build/docs/html/index.html
```

`ci/docs.sh` stops at the first warning. Its steps, each of which also runs in CI
(`.github/workflows/docs.yml`):

1. **Doxygen** on the public headers with warnings as errors, then `ci/doxygen_coverage.py`
   (the conventions of PLAN Section 9.1). The XML goes to `build/doxygen/xml`.
2. **Sphinx** (`sphinx-build -W --keep-going -n`): every page must be in a toctree, every
   cross-reference and every C++ name must resolve. Output: `build/docs/html`.
3. **Link check** (`sphinx-build -b linkcheck`) of internal links: documents, anchors and local
   files. External links are checked only with `DYNG_LINKCHECK_EXTERNAL=1`, so the gate never
   depends on the network.

Options: `ci/docs.sh --doxygen-only` runs step 1 (what `lint.yml`, the CMake target
`docs-doxygen` and Read the Docs' pre-build use); `--no-linkcheck` skips step 3. The build
directory is `DYNG_BUILD_DIR` (default `build/cpu-only`), the output `DYNG_DOCS_OUTPUT`
(default `build/docs`).

How to write pages (MyST Markdown, where things go, the C++ reference) is in
{doc}`../developer/documentation`.
