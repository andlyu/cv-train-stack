---
name: pipeline
description: |
  Turn ad-hoc processes into sharp, repeatable pipelines. Reviews a messy first-run
  script or workflow, documents it properly, adds validation, error handling, and
  makes the second run reliable. Use when: "make this repeatable", "sharpen this pipeline",
  "document this process", "pipeline review", "make this production-ready",
  "I ran this once and it worked, now make it solid".
  Proactively suggest when the user has a working but messy multi-step process.
---

# Pipeline Skill

Turn a messy first run into a sharp, repeatable pipeline. The goal: if you ran
something once and it worked, this skill makes sure the second run (and every run
after) is clean, documented, and reliable.

## Philosophy

First runs are exploratory — you're figuring out what works. But once something
works, the next step is to lock it down so it runs the same way every time without
you having to remember the details. This skill bridges that gap.

## Subcommands

Parse the user's input to determine which subcommand to run:

- `/pipeline review` → **Review** an existing script or process for repeatability
- `/pipeline sharpen` → **Sharpen** a working process into a clean pipeline
- `/pipeline document` → **Document** an existing pipeline with runbook-style docs
- `/pipeline validate` → **Validate** a pipeline handles edge cases and failures

Default to `sharpen` if no subcommand given.

---

## Review

Audit an existing script or multi-step process for repeatability issues.

### Step 1: Understand the Process

Read all relevant files. Ask the user to walk through what they did if there's no
script yet. Identify:

- **Inputs**: What data/files/configs does this need?
- **Outputs**: What does it produce?
- **Dependencies**: What tools, packages, hardware, or services are required?
- **Environment**: Where does it run? (local, remote, GPU box, Jetson, etc.)
- **State**: Does it depend on prior state? Does it modify shared state?

### Step 2: Flag Repeatability Risks

Check for these common issues:

| Risk | Example |
|------|---------|
| **Hardcoded paths** | `/home/andrew/data/batch_3/` instead of a parameter |
| **Missing validation** | No check that input files exist before processing |
| **Implicit ordering** | Steps that must run in sequence but aren't enforced |
| **Silent failures** | Errors swallowed, partial output treated as success |
| **Manual steps** | "Then I copied the files over and renamed them" |
| **Environment assumptions** | Assumes conda env, GPU, specific disk layout |
| **Non-idempotent** | Running twice corrupts data or duplicates results |
| **Missing cleanup** | Temp files, processes, or resources left behind |
| **Undocumented parameters** | Magic numbers, thresholds, or flags buried in code |
| **No progress tracking** | Can't tell if it's 10% or 90% done |

### Step 3: Report

Output a structured review:

```
PIPELINE REVIEW — <name>

Inputs:  <list>
Outputs: <list>
Runs on: <environment>

Repeatability Score: X/10

Issues:
1. [CRITICAL] ...
2. [WARNING] ...
3. [SUGGESTION] ...

Recommendation: <sharpen / document / rewrite>
```

---

## Sharpen

Take a working process and make it production-ready. This is the main operation.

### Step 1: Audit (same as Review Steps 1-2)

### Step 2: Define the Interface

Every pipeline needs a clear interface. Define:

```python
# What the user controls
INPUTS = {
    "data_dir": "Path to input data",
    "output_dir": "Where results go",
    "batch_name": "Name for this run",
    # ...
}

# What the pipeline produces
OUTPUTS = {
    "results/": "Processed output files",
    "logs/": "Run logs with timestamps",
    "summary.json": "Run metadata and stats",
}
```

Turn hardcoded values into CLI arguments with sensible defaults.

### Step 3: Add Guards

Add validation at the boundaries:

- **Pre-flight checks**: Verify inputs exist, dependencies are available, environment
  is correct, disk space is sufficient, no stale locks exist
- **Idempotency**: Either skip already-processed items, or require `--force` to overwrite
- **Atomic writes**: Write to temp files, rename on success
- **Progress tracking**: Log what's been processed so a restart can resume

### Step 4: Add Structure

Wrap the process in a clear structure:

```python
def main():
    args = parse_args()           # CLI interface with defaults
    validate_inputs(args)         # Pre-flight checks
    setup_logging(args)           # Structured logging
    
    with RunContext(args) as ctx:  # Tracks run metadata
        results = process(ctx)    # The actual work
        ctx.save_summary(results) # Persist run stats
    
    print_summary(results)        # Human-readable output
```

### Step 5: Add a Runbook Header

Every pipeline script should have a docstring or header comment that serves as a
quick-reference runbook:

```python
"""
<Pipeline Name>

What: <one-line description>
When: <when to run this>
How:  <the command to run>

Prerequisites:
  - <what needs to be true before running>

Common variations:
  - <flag>: <what it does>

Output:
  - <what gets produced and where>

Example:
  python pipeline.py --data-dir ./batch_4 --output-dir ./results
"""
```

### Step 6: Implement

Make the changes. Prefer editing the existing script over creating a new one.
Keep the core logic intact — wrap it, don't rewrite it.

---

## Document

Generate runbook-style documentation for an existing pipeline that already works
well but lacks docs.

### Output Format

Create a `PIPELINE.md` or add a docstring with:

1. **Purpose** — What does this do and why?
2. **Prerequisites** — What needs to be installed/configured/available?
3. **Quick Start** — Copy-paste command to run the common case
4. **Parameters** — Table of all flags/arguments with defaults
5. **Input/Output** — What goes in, what comes out, where
6. **Common Variations** — The 2-3 most common ways to run it
7. **Troubleshooting** — Known failure modes and fixes
8. **History** — When it was created and major changes

---

## Validate

Test that a pipeline handles edge cases gracefully.

### Checks to Run

1. **Missing inputs** — What happens if input files don't exist?
2. **Empty inputs** — What happens with zero items to process?
3. **Partial failure** — What happens if step 3 of 5 fails?
4. **Re-run** — What happens if you run it twice on the same data?
5. **Interrupt** — What happens if you Ctrl-C mid-run?
6. **Disk full** — Does it check disk space before writing?
7. **Concurrent runs** — What happens if two instances run at once?

Report which checks pass/fail and suggest fixes for failures.

---

## Principles

1. **Don't rewrite working code** — Wrap it, add guards, add docs. The core logic
   already works; don't introduce new bugs by refactoring.

2. **CLI args with good defaults** — Every knob should be a flag, but the default
   should be the common case. `python pipeline.py` should just work.

3. **Fail fast, fail loud** — Check preconditions upfront. Don't process 90% of
   the data before discovering the output dir doesn't exist.

4. **Idempotent by default** — Running twice should be safe. Skip or warn on
   already-processed items rather than corrupting them.

5. **Log everything, print summaries** — Detailed logs go to files; the terminal
   gets a clean summary of what happened.

6. **Atomic outputs** — Write to temp, rename on success. Never leave partial
   output that looks complete.

7. **Resume over restart** — If a pipeline processes 1000 items and fails at 750,
   the next run should pick up at 751, not start over.
