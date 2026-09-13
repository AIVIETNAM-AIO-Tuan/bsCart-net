## Engineering & Research Discipline

### 1. Failure-First Validation

* Never declare a change successful because the primary path works once.
* Before concluding, actively identify and test the maximum plausible failure modes within the available context, tools, and compute.
* Treat temporary success, a single passing run, or visually plausible output as insufficient evidence.
* Check edge cases, invalid inputs, missing data, shape/type mismatches, numerical instability, silent failures, unexpected state, and downstream incompatibilities when relevant.
* Distinguish between:

  * **verified working**
  * **likely working but unverified**
  * **known limitation**
* Never hide unresolved failures behind fallbacks unless the fallback is intentionally part of the design.

### 2. System-Wide Impact Analysis

* Every code change, including a one-line change, must be evaluated at three levels:

  1. local function/file,
  2. owning module and its callers/dependencies,
  3. end-to-end pipeline behavior.
* Before editing shared interfaces, configs, tensors, schemas, losses, preprocessing, checkpoints, or data formats, inspect all known consumers and producers.
* Never optimize one component while silently degrading another stage.
* After meaningful changes, check for regressions in related modules and pipeline assumptions.
* For cross-module impact analysis, prefer Graphify graph traversal over manual grep.

### 3. Definition of Done

A task is not done until all applicable conditions are satisfied:

* implementation is complete,
* relevant failure modes were considered,
* affected dependencies were inspected,
* outputs were validated,
* regressions were checked,
* obsolete code/artifacts were removed or clearly deprecated,
* documentation/configuration was updated when behavior changed,
* Graphify was updated after code modification.

Do not stop merely because the immediate error disappeared.

### 4. Repository Architecture

* Maintain the repository as an engineer-facing codebase, not as a collection of temporary scripts.
* Do not create arbitrary notebooks, scripts, folders, or duplicated implementations at repository root.
* Organize code by stable responsibility and research module, not by chronological convenience.
* Each major research direction or improvement should have a clear module boundary and ownership.
* Prefer structures such as:

```text
src/
  data/
  preprocessing/
  models/
    segmentation/
    registration/
    synthesis/
    fusion/
  losses/
  training/
  evaluation/
  utils/

experiments/
  <research_direction>/
    configs/
    scripts/
    notebooks/
    outputs/
    README.md

tests/
configs/
docs/
```

* Adapt the exact structure to the project; do not force folders that have no real responsibility.
* Experimental notebooks belong under the corresponding experiment/research direction.
* Reusable logic must live in modules, not remain duplicated inside notebooks.
* A notebook should primarily orchestrate, visualize, or analyze; it should not become the only implementation of core pipeline logic.
* Do not create a new implementation if an existing module can be cleanly extended.
* Avoid orphan files whose role, owner, or relation to the pipeline is unclear.

### 5. Research Direction Isolation

* Keep competing research ideas separable.
* Do not entangle experimental branches conceptually inside core modules before they are validated.
* Prefer configuration-driven variants over duplicated pipelines.
* Clearly distinguish:

  * baseline,
  * current proposed method,
  * ablation,
  * exploratory prototype,
  * deprecated approach.
* Experimental changes must be reversible and attributable to a specific hypothesis.

### 6. Reproducibility

* Every significant experiment must be reproducible from repository state plus configuration.
* Record relevant dataset version/split, preprocessing, model configuration, losses, seeds, checkpoint source, and evaluation protocol.
* Avoid hidden parameters inside notebooks or interactive sessions.
* Prefer versioned configuration files over manually edited constants.
* Never compare experiments whose evaluation conditions differ without explicitly stating the difference.

### 7. Experimental Integrity

* Do not infer improvement from isolated examples.
* Separate qualitative observations from quantitative evidence.
* When reporting improvement, inspect whether gains may come from leakage, preprocessing differences, easier subsets, changed evaluation protocol, random variance, or implementation bugs.
* Preserve meaningful negative results; failed hypotheses are research information, not disposable noise.
* Never modify the baseline in a way that makes comparison unfair without explicitly documenting it.

### 8. Major Research Events — `Event.md`

For every major research, architecture, methodology, or experiment-design change, append an entry to `Event.md`.

A major event includes, but is not limited to:

* architecture redesign,
* addition/removal of a major module,
* change of research hypothesis,
* dataset or evaluation protocol change,
* major loss/objective modification,
* major synthetic-data strategy change,
* abandoning or adopting a research direction,
* significant experimental result,
* user decision that changes subsequent research direction.

Each entry must record:

```markdown
## YYYY-MM-DD — <Event title>

### Context
Why this change/question arose.

### Change
What changed technically or methodologically.

### Evidence / Result
Observed quantitative or qualitative result.
Use "Not yet evaluated" when applicable.

### Significance
Why the result or change matters.

### Risks / Open Questions
Remaining uncertainties or possible failure modes.

### Decision
The user's decision or agreed conclusion at that time.

### Consequence
What future implementation/experiments should follow from this decision.
```

* Do not rewrite historical decisions after later results appear.
* Add a new event that supersedes the old decision instead.
* `Event.md` is the chronological research decision record.

### 9. Daily Progress Record — `Progress.md`

At the end of each working day, update `Progress.md`.

Record only information useful for continuing the project:

```markdown
## YYYY-MM-DD

### Completed
Important implementation, analysis, experiments, fixes, or documentation completed today.

### Critical Changes
Changes that affect architecture, interfaces, data flow, assumptions, or future work.

### Findings
Important positive, negative, or unexpected findings.

### Failures / Risks
Known bugs, failed experiments, fragile assumptions, unresolved issues, or suspected regressions.

### Decisions
Important decisions made today. Reference `Event.md` when the decision is a major research event.

### Current State
What is currently working, partially working, or unverified.

### Next
Highest-priority next actions.
```

* Do not turn `Progress.md` into a verbose chat log.
* Preserve enough context that another engineer can resume work without reconstructing the entire day.
* Never mark an unresolved issue as completed merely because a workaround exists.

### 10. Decision and Assumption Traceability

* Important assumptions must be explicit.
* When implementation depends on an uncertain research assumption, mark it as such in code/config/docs rather than presenting it as established fact.
* When a later change invalidates an earlier assumption, identify affected modules and experiments.
* Significant user decisions override previous speculative directions and must be reflected in `Event.md`.

### 11. Cleanup and Technical Debt

* Do not accumulate obsolete experimental files after a direction is superseded.
* Before deleting potentially useful research artifacts, verify whether they are referenced by `Event.md`, `Progress.md`, experiments, configs, or reproducibility records.
* Clearly deprecate code that cannot yet be safely removed.
* Do not leave commented-out alternative implementations as permanent history; history belongs in version control and research records.

### 12. Graphify Discipline

* Before answering architecture or codebase questions, read `graphify-out/GRAPH_REPORT.md` for god nodes and community structure.
* If `graphify-out/wiki/index.md` exists, navigate it instead of reading raw files first.
* For cross-module relationship questions, prefer:

  * `graphify query "<question>"`
  * `graphify path "<A>" "<B>"`
  * `graphify explain "<concept>"`
* Use raw file inspection only when implementation-level details are required after graph-level navigation.
* After modifying any code file, run:

```bash
graphify update .
```

* When assessing the impact of a change, use Graphify to inspect upstream dependencies, downstream consumers, and relevant inferred relationships whenever possible.

### 13. Core Principle

Optimize for **correctness, traceability, reproducibility, maintainability, and research validity**, not merely for making the current command pass.

A locally successful change that introduces hidden pipeline risk is a failed change.
