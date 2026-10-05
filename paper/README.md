# Paper evidence and reproduction

This directory maps the manuscript's tests to their retained evidence and provides
checks for the **external companion study package**. It contains metadata and check
code only. It does not include the manuscript, raw observations, model KIs, binaries,
campaign histories or model environments. Obtain the companion package from the
study authors; no public download location is supplied here.

From the repository root:

```bash
python paper/reproduce.py list
python paper/reproduce.py list --test E4_VIC_PAIRED
python paper/reproduce.py validate --paper-root /path/to/GRL_SUBMISSION_PACKAGE_2026-09-24
python paper/reproduce.py check-saved --paper-root /path/to/GRL_SUBMISSION_PACKAGE_2026-09-24 --output ./outputs/saved.json
python paper/reproduce.py environments --paper-root /path/to/GRL_SUBMISSION_PACKAGE_2026-09-24 --case FSM2 --output ./outputs/environment.json
python paper/reproduce.py figure-s2 --paper-root /path/to/GRL_SUBMISSION_PACKAGE_2026-09-24 --preflight
python paper/reproduce.py figure-s2 --paper-root /path/to/GRL_SUBMISSION_PACKAGE_2026-09-24 --output-dir ./outputs/figure-s2
```

Use Python 3.12 or newer. Listing, index validation, environment checks and Figure S2
preflight use the standard library. GR4J saved-contract scoring requires PyYAML;
rendering Figure S2 requires NumPy and Matplotlib:

```bash
python -m pip install -r paper/requirements-checks.txt
```

| Command | What a successful check establishes |
|---|---|
| `list` | Shows test scope, evidence status and archive-relative paths; no archive needed. Navigation entries include subtests and dependencies, not independent experiment counts. |
| `validate` | Four index versions match the recorded source hashes; paper/test relationships and all indexed file paths resolve inside the companion package. It does not hash every payload or certify scientific correctness. |
| `check-saved` | Six complete reports contain 23 components and pass the historical consistency check; current GR4J contracts reproduce the archived comparison; eight selector implementation tests pass. Select one using `--check six_model`, `gr4j_current` or `selector_unit`. Archived acceptance flags are not independently recomputed. |
| `environments` | Selected case manifests and native binary identities match, with host/setup conditions reported separately. Repeat `--case` or omit it to inspect all twelve environments. File integrity does not establish execution readiness. |
| `figure-s2` | The plotted 23 component records match the six saved reports. `--preflight` imports no figure builder; rendering invokes only the archived Figure S2 function in a temporary copy. Outputs include PNG/PDF/SVG and a provenance manifest. |

These commands do not run physical models or contact language-model providers.
`check-saved` executes the relevant archived scorer and implementation tests; use a
trusted study package. Outputs must be outside the companion package. The figure
builder refuses to replace existing exports. A nonzero exit means missing inputs,
a failed check or a configuration error; environment exit zero still does not mean
that a numerical run is ready.

The original paper used its frozen toolkit, not necessarily this repository's current
engine. For numerical reruns use the companion package's
`03_code/Recovered_Dependencies/Environment_Setup/README_SETUP.md` and each case's
instructions. The exact sealed GR4J Python runtime is unrecovered. Cross-provider
authoring retains console evidence and reconstructed scripts; the additional nine
authored contracts are unavailable. Neither gap is filled by these checks.

The companion archive also contains restricted observations and third-party model
material. It is not a ready-to-publish dataset: follow its per-case source/access notes
and `04_research_archive/record/LICENSES_AND_THIRD_PARTY.md`. The GitHub metadata is
only a navigation map, not a redistribution of those inputs.

`metadata/SOURCE_MANIFEST.json` identifies the original index/check hashes, package
version and metadata export. Absolute provenance fields and historical shell command
blocks were omitted from the copied indexes; scientific evidence labels and
package-relative locators are retained. Paths are resolved only from `--paper-root`,
never from a historical machine path. Adapted checks are derived from the study
authors' package scripts; the source manifest records their originals.
