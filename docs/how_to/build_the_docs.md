# Build the documentation

The site is built from the Markdown pages in `docs/` and the Doxygen comments of the public
headers, with the tools pinned in `environment.yml` (Sphinx, MyST, pydata-sphinx-theme,
Breathe, Doxygen).

```bash
source scripts/dev_env.sh
cmake --preset cpu-only       # once: generates version.hpp / config.hpp for Doxygen
ci/docs.sh                    # Doxygen + checks, the Sphinx site (-W), the link check
xdg-open build/docs/html/index.html
```

`ci/docs.sh` stops at the first warning. Its steps, each of which also runs in CI
(`.github/workflows/docs.yml`):

1. **Doxygen** on the public headers with warnings as errors, then `ci/doxygen_coverage.py`
   (the Doxygen conventions of CONTRIBUTING.md). The XML goes to `build/doxygen/xml`.
2. **Sphinx** (`sphinx-build -W --keep-going -n`): every page must be in a toctree, every
   cross-reference and every C++ name must resolve. This is also the check of the links between
   pages: a Markdown link or `{doc}` to a missing page, heading or local file is an error.
   Output: `build/docs/html`.
3. **Link check** (`ci/docs_links.py`) of what Sphinx cannot see: every
   `https://github.com/dyng-dev/dyng/blob/main/<path>` (or `tree/main/...`) link in the
   repository's files must name an existing path, and every relative link of the built pages an
   existing page and `#anchor`. External links (other sites) are checked only with
   `DYNG_LINKCHECK_EXTERNAL=1` (Sphinx's linkcheck builder; the weekly job of `docs.yml`), so the
   gate never depends on the network.

Options: `ci/docs.sh --doxygen-only` runs step 1 (what the CMake target `docs-doxygen` and
Read the Docs' pre-build use); `--no-linkcheck` skips step 3. The build
directory is `DYNG_BUILD_DIR` (default `build/cpu-only`), the output `DYNG_DOCS_OUTPUT`
(default `build/docs`).

From CMake: configure with `-DDYNG_BUILD_DOCS=ON` (it needs doxygen and sphinx-build) and the
target `docs`, part of `all`, runs `ci/docs.sh` for that build tree, writing the site to
`<build>/docs/html`: `cmake --preset cpu-only -DDYNG_BUILD_DOCS=ON && cmake --build --preset
cpu-only --target docs`.

How to write pages (MyST Markdown, where things go, the C++ reference) is in
{doc}`../developer/documentation`.
