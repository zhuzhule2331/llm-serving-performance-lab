# License note

This repository mixes **original code**, **measured results**, and **third-party dependencies**.
The three are not covered by the same terms, so no single SPDX identifier is asserted here.

## 1. Original code and documentation

The contents of `common/`, `experiments/`, `scripts/`, `figures/make_figures.py`, `docs/`, and
`README.md` were written by the repository author.

**No license file is included yet.** Until one is added, default copyright applies and the code
is *not* licensed for reuse. If you are the author and intend to publish this publicly, pick one
deliberately — for a portfolio repository of this shape, `MIT` or `Apache-2.0` are the usual
choices:

- **MIT** — shortest, most permissive, most common for portfolio/reference code.
- **Apache-2.0** — same permissiveness plus an explicit patent grant; slightly heavier.

Adding a `LICENSE` file is a deliberate decision and was **not** made automatically here.

## 2. Measured results

`results/*.csv` and `results/*.txt` are **measurements produced by the author** on the author's
hardware. They are published as evidence, not as a dataset with separate terms. If you reuse the
numbers, cite this repository.

## 3. Third-party components

| Component | Role | Terms |
|---|---|---|
| **vLLM** | the serving engine under test | Apache-2.0 |
| **Qwen3-VL-8B-Instruct** | the model under test | see the model card on Hugging Face; **weights are not redistributed here** |
| **NVIDIA Nsight Systems** | profiler used in Exp12 | NVIDIA proprietary; **not redistributed**; raw captures not published |
| **openai (Python SDK)** | HTTP client for the OpenAI-compatible endpoint | Apache-2.0 |
| **matplotlib**, **pandas**, **numpy** | analysis and figures | BSD-style / BSD-3 / BSD-3 |

No third-party binaries, model weights, or profiler captures are included in this repository.

## 4. Trademarks

"vLLM", "Qwen", "NVIDIA", "Nsight", "CUDA", and "TensorRT" are trademarks of their respective
owners. Their use here is descriptive and does not imply endorsement or affiliation.
