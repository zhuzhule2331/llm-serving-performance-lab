# Quality Evaluation

The performance work is only meaningful if the INT8 path still produces usable model
output. This document records what was measured, what the numbers are, and — just as
importantly — **what the design cannot support as a conclusion**.

---

## 1. Design

| Component | Detail |
|---|---|
| Model | `Qwen3-VL-8B-Instruct` |
| Comparators | BF16 LM Head vs INT8 LM Head, same server flags otherwise |
| Objective set | 22 questions, automatic grading |
| Open-ended set | 13 deduplicated questions, single blind human review (0–4 per question) |
| Sampling | `temperature=0` |
| Generation budget | 384 tokens for the original pass; 768 tokens for a targeted Q026 retest; longer budget for the open-ended retest |
| Blinding | answers anonymised as X/Y before scoring; unblinded afterwards by **exact text matching** against the stored model outputs |

Only **one** review pass was performed. There is no inter-rater agreement measure.

---

## 2. Objective set — 22 questions

### 2.1 Result

| Category | n | BF16 | INT8 |
|---|---:|---:|---:|
| 基本计算能力 (basic computation) | 5 | 5 | 5 |
| 基本推理思考能力 (basic reasoning) | 5 | 2 | 2 |
| 进阶解题能力 (advanced problem solving) | 3 | 3 | 3 |
| 进阶计算能力 (advanced computation) | 5 | 4 | 4 |
| 进阶推理能力 (advanced reasoning) | 4 | 4 | 3 |
| **Total** | **22** | **18** | **17** |

Machine-readable source: `artifacts/quality_ab/objective_summary.json`.
Per-question outputs: `artifacts/quality_ab/full_results.json`.

### 2.2 The single-point difference is a truncation artefact

**Q026** (advanced reasoning): the BF16 run finished normally; the INT8 run reached the
384-token limit before producing a parseable answer.

A **768-token targeted retest** (`retest_q026.py`) was run afterwards:

| Model | Correct rounds |
|---|---|
| BF16 | 3 / 3 |
| INT8 | 3 / 3 |

The retest is stored in `artifacts/quality_ab/q026_retest_768.json`. **It does not overwrite
the original 384-token score** — both are reported.

### 2.3 Two categories are not pure knowledge accuracy

Q011 and Q015 failed automatic parsing for **both** models because both emitted a bare
`答案：结果` as the final line. The prose before that line pointed in the right direction.
The automatic score therefore measures *parseability under a strict rule* as much as
knowledge. Q012 and Q023 show reasoning/computation errors that are **shared by both
models**, not INT8-specific.

---

## 3. Open-ended set — 13 deduplicated questions

### 3.1 Result

| Slice | BF16 | INT8 |
|---|---:|---:|
| Original answers, 13 questions | 37/52 | 38/52 |
| Long-answer retest, 8 questions | 22/32 | 20/32 |
| **Deduplicated 13 questions (headline)** | **41/52** (78.85%) | **39/52** (75.00%) |

The two slices overlap on 8 question IDs and **must not be added together** into 21
independent questions.

### 3.2 Where the two-point gap comes from

| Question | BF16 | INT8 | Note |
|---|---:|---:|---|
| Q032 | 4 | 3 | both explain prefill/decode/TTFT/TPOT; INT8 answer stops at the 1200-token limit (`finish_reason=length`) vs BF16 `stop` at 1195 tokens |
| Q033 | 3 | 2 | both give per-row quantization / `scale = max_abs/127` / `W ≈ q × scale`; INT8 adds an inaccurate "scale mismatch" risk framing |
| **all other 11** | — | — | identical scores |

The 5-token output-length difference in Q032 is **not** itself evidence of capability.

### 3.3 Questions where both models were weak

- **Q030 (2 / 2)**: both recognise that unequal token counts confound comparison, but
  neither runs a same-GPU / swapped-GPU crossover. INT8 additionally miscalculates
  `0.95 × (100/92)` as 96.7% (correct ≈103.26%).
- **Q034 (1 / 1)**: both confuse UVA, pinned memory and mapped pinned memory. This is a
  **shared** limitation, not an INT8 artefact.

### 3.4 What each file actually contains

| File | What it is | Scored? |
|---|---|---|
| `artifacts/quality_ab/blind_review.json` | the **blind packet**: 13 questions with anonymised `X` / `Y` answers, criteria and notes | ❌ **No** — `score_X_0_to_4` and `score_Y_0_to_4` are `null` in all 13 rows |
| `artifacts/quality_ab/blind_review_unblinded_scores.csv` | the **post-scoring sheet**: 21 rows (13 original + 8 long-retest), with `X_identity` / `Y_identity`, `BF16_score` / `INT8_score`, finish reason, token counts and reviewer notes | ✅ Yes |

The blind packet and the score sheet are **two different files**. `blind_review.json` must
not be described as "the scored file" — it contains no scores.

### 3.5 Unblinding and deduplication (how 41/52 and 39/52 are computed)

Unblinding used **exact string equality**: each anonymised answer was matched against the
stored BF16 / INT8 outputs in `full_results.json` and `retest_open_long_full.json`.
Labels, scores and answer positions were not used to infer identity. For Q001–Q004 the two
answers are byte-identical, so identity is not recoverable from text alone — but both were
scored the same, so the total is unaffected.

Deduplication rule (13 questions, 52 points max):

- **Q001–Q005** → use the `original_excerpt` row
- **Q019, Q020, Q030, Q031, Q032, Q033, Q034, Q035** → use the `long_retest` row

This selects exactly these rows from `blind_review_unblinded_scores.csv`:

| ID | BF16 | INT8 | ID | BF16 | INT8 |
|---|---:|---:|---|---:|---:|
| Q001 | 4 | 4 | Q031 | 2 | 2 |
| Q002 | 4 | 4 | Q032 | 4 | 3 |
| Q003 | 4 | 4 | Q033 | 3 | 2 |
| Q004 | 3 | 3 | Q034 | 1 | 1 |
| Q005 | 4 | 4 | Q035 | 2 | 2 |
| Q019 | 4 | 4 | | | |
| Q020 | 4 | 4 | | | |
| Q030 | 2 | 2 | | | |
| **subtotal (Q001–Q030)** | **33** | **33** | **subtotal (Q031–Q035)** | **8** | **7** |
| | | | **total** | **41 / 52** | **39 / 52** |

Reproduce with:

```bash
python - <<'PY'
import csv, io
rows = list(csv.DictReader(io.StringIO(
    open("artifacts/quality_ab/blind_review_unblinded_scores.csv", encoding="utf-8-sig").read())))
sel = [r for r in rows
       if (r["dataset"] == "original_excerpt" and r["id"] in {"Q001","Q002","Q003","Q004","Q005"})
       or (r["dataset"] == "long_retest" and r["id"] in {"Q019","Q020","Q030","Q031","Q032","Q033","Q034","Q035"})]
print(len(sel), sum(int(r["BF16_score"]) for r in sel), sum(int(r["INT8_score"]) for r in sel))
PY
# -> 13 41 39
```

> **Statistical status:** these are **one subjective review pass** of truncated
> answers, scored 0–4 per question. There is no second rater, no inter-rater agreement, no
> confidence interval and no hypothesis test. **41/52 vs 39/52 must not be described as a
> statistically significant quality difference.** Two points out of 52 come entirely from
> Q032 and Q033; the other 11 questions scored identically.

---

## 4. What may and may not be claimed

| ✅ Supported | ❌ Not supported |
|---|---|
| On this 22-question set, objective scores were BF16 18/22 and INT8 17/22 | "INT8 is 4.5% worse than BF16" |
| The one-point gap is attributable to a truncated Q026 answer, which both models answered correctly with a larger budget | "the quality difference is zero" |
| On a 13-question deduplicated open set, one blind reviewer scored BF16 41/52 and INT8 39/52 | "quality degradation is statistically proven" |
| INT8 output is **not identical** to BF16 | "lossless" / "严格无损" |
| The observed differences are small and question-dependent | "INT8 produces the same answers" |

The evaluation is a **sanity check**, not a quality study. It has one reviewer, truncated
answers, one to a few generations per question, and no confidence intervals.

---

## 5. Reproducing

```bash
# requires BF16 server on :8001 and INT8 server on :8002
python scripts/quality_ab_suite.py
python scripts/retest_q026.py
python scripts/retest_open_truncated.py
```

`quality_ab_suite.py` writes `artifacts/quality_ab/full_results.json`. The retest scripts
refuse to overwrite existing result files, by design — the original scores are treated as
immutable.

The blind scores were produced by a human reviewer and are stored in
`artifacts/quality_ab/blind_review_unblinded_scores.csv`. **They are not regenerable by
script.** The separate `blind_review.json` is the **unscored** packet the reviewer worked
from.
