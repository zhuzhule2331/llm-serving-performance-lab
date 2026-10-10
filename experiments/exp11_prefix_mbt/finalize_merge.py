# =============================================================================
# Source  : llm_serving_portfolio_20261008/raw_bench/exp11_prefix_mbt/exp11_finalize_merge.py
# Copied  : verbatim except for the changes listed in ../../PUBLIC_CODE_CHANGELOG.md
# Frozen  : this file is the FINAL public-facing implementation used in the
#           formal campaign; it is a copy, the original is unmodified.
# =============================================================================
#!/usr/bin/env python3

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent

ORIGINAL = (
    ROOT
    / "exp11_formal_20261008_175541"
)

REPLACEMENT = (
    ROOT
    / "exp11_block2_replacement_20261008_185659"
)

RECOVERY_POLICY = (
    ROOT
    / "exp11_block2_recovery_policy_v1.json"
)

RECOVERY_POLICY_HASH = (
    ROOT
    / "exp11_block2_recovery_policy_v1_SHA256.txt"
)


# Original INVALID attempt was already frozen.
ORIGINAL_EXPECTED_HASHES = {
    "manifest.json":
        "eaeafdd7e47f31c72da39119d290418e0bd0c2962532ddec9ba15150d61a976a",

    "formal_bursts_progress.csv":
        "5c8efed20096527e51650998228b19d6d6cd63d7b9799d488da6a5b082f60629",

    "formal_requests_progress.csv":
        "9e46f0ecce2c88b56b44227e45a4da2cca465b1509c9f1243a72731cee24b23c",

    "block_status.csv":
        "ebc0e5d9728fb2db0e875215d8d42fe1be089ab894acba1bcc7119cb71d3be35",
}


def sha256_file(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def truth(x):

    if isinstance(x, bool):
        return x

    return str(x).lower() == "true"


def read_csv(path):

    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:

        return list(
            csv.DictReader(f)
        )


def write_csv(path, rows):

    if not rows:
        raise RuntimeError(
            f"Refusing to write empty CSV: {path}"
        )

    fields = []

    for row in rows:

        for key in row.keys():

            if key not in fields:
                fields.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


def verify_original():

    for name, expected in (
        ORIGINAL_EXPECTED_HASHES.items()
    ):

        actual = sha256_file(
            ORIGINAL / name
        )

        if actual != expected:

            raise RuntimeError(
                "ORIGINAL ARTIFACT HASH "
                f"MISMATCH: {name}\n"
                f"expected={expected}\n"
                f"actual={actual}"
            )

    print(
        "ORIGINAL INVALID ATTEMPT HASHES: PASS"
    )


def verify_replacement():

    hash_path = (
        REPLACEMENT
        / "replacement_artifact_hashes.json"
    )

    expected = json.loads(
        hash_path.read_text()
    )

    for name, digest in (
        expected.items()
    ):

        actual = sha256_file(
            REPLACEMENT / name
        )

        if actual != digest:

            raise RuntimeError(
                "REPLACEMENT ARTIFACT HASH "
                f"MISMATCH: {name}\n"
                f"expected={digest}\n"
                f"actual={actual}"
            )

    print(
        "REPLACEMENT ARTIFACT HASHES: PASS"
    )


def verify_recovery_policy():

    expected = (
        RECOVERY_POLICY_HASH
        .read_text()
        .strip()
        .split()[0]
    )

    actual = sha256_file(
        RECOVERY_POLICY
    )

    if actual != expected:

        raise RuntimeError(
            "RECOVERY POLICY HASH MISMATCH"
        )

    print(
        "RECOVERY POLICY HASH: PASS"
    )

    return actual


def add_source(
    rows,
    source_attempt,
    source_dir,
):

    output = []

    for original in rows:

        row = dict(original)

        row[
            "formal_source_attempt"
        ] = source_attempt

        row[
            "formal_source_dir"
        ] = str(source_dir)

        output.append(row)

    return output


def main():

    verify_original()
    verify_replacement()

    recovery_hash = (
        verify_recovery_policy()
    )

    original_bursts = read_csv(
        ORIGINAL
        / "formal_bursts_progress.csv"
    )

    original_requests = read_csv(
        ORIGINAL
        / "formal_requests_progress.csv"
    )

    replacement_bursts = read_csv(
        REPLACEMENT
        / "replacement_all_executions.csv"
    )

    replacement_requests = read_csv(
        REPLACEMENT
        / "replacement_formal_requests.csv"
    )

    # --------------------------------------------------------
    # Select ONLY original B1 and B3.
    # Original B2 is permanently excluded.
    # --------------------------------------------------------

    original_kept_bursts = [
        r
        for r in original_bursts
        if int(r["block_id"])
        in {1, 3}
    ]

    original_kept_requests = [
        r
        for r in original_requests
        if (
            int(r["block_id"])
            in {1, 3}
            and
            truth(
                r[
                    "include_in_formal"
                ]
            )
        )
    ]

    if len(
        original_kept_bursts
    ) != 54:

        raise RuntimeError(
            "Expected 54 total original "
            "B1+B3 executions, got "
            f"{len(original_kept_bursts)}"
        )

    if len(
        original_kept_requests
    ) != 288:

        raise RuntimeError(
            "Expected 288 measured request "
            "rows from original B1+B3, got "
            f"{len(original_kept_requests)}"
        )

    # --------------------------------------------------------
    # Replacement MUST be complete Block 2 only.
    # --------------------------------------------------------

    if len(
        replacement_bursts
    ) != 27:

        raise RuntimeError(
            "Replacement burst count != 27"
        )

    if not all(
        int(r["block_id"]) == 2
        for r in replacement_bursts
    ):

        raise RuntimeError(
            "Replacement contains "
            "non-Block-2 rows"
        )

    if len(
        replacement_requests
    ) != 144:

        raise RuntimeError(
            "Replacement measured request "
            "count != 144"
        )

    if not all(
        int(r["block_id"]) == 2
        for r in replacement_requests
    ):

        raise RuntimeError(
            "Replacement request file "
            "contains non-Block-2 rows"
        )

    # --------------------------------------------------------
    # Add explicit provenance.
    # --------------------------------------------------------

    all_executions = (
        add_source(
            original_kept_bursts,
            "original_v1",
            ORIGINAL,
        )
        +
        add_source(
            replacement_bursts,
            "block2_replacement_v1",
            REPLACEMENT,
        )
    )

    formal_bursts = [
        r
        for r in all_executions
        if truth(
            r["include_in_formal"]
        )
    ]

    formal_requests = (
        add_source(
            original_kept_requests,
            "original_v1",
            ORIGINAL,
        )
        +
        add_source(
            replacement_requests,
            "block2_replacement_v1",
            REPLACEMENT,
        )
    )

    # --------------------------------------------------------
    # Canonical order.
    # --------------------------------------------------------

    all_executions.sort(
        key=lambda r:
            int(
                r["formal_row_id"]
            )
    )

    formal_bursts.sort(
        key=lambda r:
            int(
                r["formal_row_id"]
            )
    )

    formal_requests.sort(
        key=lambda r: (
            int(
                r["formal_row_id"]
            ),
            int(
                r["request_id"]
            ),
        )
    )

    # --------------------------------------------------------
    # Hard structural audits.
    # --------------------------------------------------------

    if len(all_executions) != 81:

        raise RuntimeError(
            "Final total execution count "
            f"!= 81: {len(all_executions)}"
        )

    if len(formal_bursts) != 54:

        raise RuntimeError(
            "Final measured burst count "
            f"!= 54: {len(formal_bursts)}"
        )

    if len(formal_requests) != 432:

        raise RuntimeError(
            "Final measured request count "
            f"!= 432: {len(formal_requests)}"
        )

    ids = [
        int(
            r["formal_row_id"]
        )
        for r in all_executions
    ]

    if ids != list(
        range(1, 82)
    ):

        raise RuntimeError(
            "Formal row IDs are not "
            "exactly 1..81"
        )

    # Explicit proof original B2 did not leak.
    for row in all_executions:

        block = int(
            row["block_id"]
        )

        source = row[
            "formal_source_attempt"
        ]

        if (
            block == 2
            and
            source
            != "block2_replacement_v1"
        ):

            raise RuntimeError(
                "ORIGINAL BLOCK 2 LEAKED "
                "INTO FINAL MERGE"
            )

        if (
            block in {1, 3}
            and
            source != "original_v1"
        ):

            raise RuntimeError(
                "B1/B3 provenance mismatch"
            )

    # All executions, including warmups,
    # must be valid.
    if not all(
        truth(
            r["measurement_valid"]
        )
        for r in all_executions
    ):

        raise RuntimeError(
            "At least one final execution "
            "is invalid"
        )

    if not all(
        float(
            r["preemption_delta"]
        ) == 0.0
        for r in all_executions
    ):

        raise RuntimeError(
            "At least one final execution "
            "has preemption != 0"
        )

    # --------------------------------------------------------
    # Design balance audit.
    # --------------------------------------------------------

    block_counts = Counter(
        int(r["block_id"])
        for r in formal_bursts
    )

    if block_counts != {
        1: 18,
        2: 18,
        3: 18,
    }:

        raise RuntimeError(
            f"Measured block imbalance: "
            f"{block_counts}"
        )

    cell_counts = Counter(
        (
            int(r["mbt"]),
            r[
                "reuse_condition"
            ],
        )
        for r in formal_bursts
    )

    expected_cells = {
        (mbt, reuse): 6
        for mbt in [
            512,
            2048,
            4096,
        ]
        for reuse in [
            "P0",
            "P50",
            "PHIGH",
        ]
    }

    if cell_counts != expected_cells:

        raise RuntimeError(
            "Final 3x3 measured cell "
            f"balance failed: {cell_counts}"
        )

    # Each measured burst must have 8 requests.
    request_counts = Counter(
        int(
            r["formal_row_id"]
        )
        for r in formal_requests
    )

    if (
        len(request_counts) != 54
        or
        set(
            request_counts.values()
        ) != {8}
    ):

        raise RuntimeError(
            "Measured request grouping "
            "is not 54 x 8"
        )

    # Exact measured workload audit.
    for r in formal_requests:

        if int(
            r["prompt_tokens"]
        ) != 2048:

            raise RuntimeError(
                "Prompt token mismatch"
            )

        if int(
            r["completion_tokens"]
        ) != 128:

            raise RuntimeError(
                "Completion token mismatch"
            )

    print(
        "FINAL STRUCTURAL VALIDITY: PASS"
    )

    print(
        "executions: 81"
    )

    print(
        "measured bursts: 54"
    )

    print(
        "measured requests: 432"
    )

    print(
        "cell balance: 9 cells x 6 bursts"
    )

    print(
        "preemption: all zero"
    )

    print(
        "original Block 2 excluded: PASS"
    )

    # --------------------------------------------------------
    # Emit immutable merged campaign directory.
    # --------------------------------------------------------

    timestamp = (
        datetime.now()
        .astimezone()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    out = (
        ROOT
        / (
            "exp11_final_merged_"
            + timestamp
        )
    )

    out.mkdir()

    write_csv(
        out
        / "all_executions.csv",
        all_executions,
    )

    write_csv(
        out
        / "formal_bursts.csv",
        formal_bursts,
    )

    write_csv(
        out
        / "formal_requests.csv",
        formal_requests,
    )

    provenance = {
        "experiment":
            "Exp11 Prefix Reuse x Scheduler Token Budget",

        "status":
            "FINAL_VALID_MERGED_CAMPAIGN",

        "created_at":
            datetime.now()
            .astimezone()
            .isoformat(),

        "block_sources": {
            "1":
                str(ORIGINAL),

            "2":
                str(REPLACEMENT),

            "3":
                str(ORIGINAL),
        },

        "excluded_source": {
            "original_block_2":
                str(ORIGINAL),

            "reason":
                (
                    "formal_row_id=48 "
                    "launch_skew hard-validity "
                    "failure in original attempt"
                ),
        },

        "recovery_policy": {
            "path":
                str(
                    RECOVERY_POLICY
                ),

            "sha256":
                recovery_hash,
        },

        "counts": {
            "all_executions":
                81,

            "measured_bursts":
                54,

            "measured_requests":
                432,
        },

        "design": {
            "blocks":
                3,

            "mbt":
                [
                    512,
                    2048,
                    4096,
                ],

            "reuse":
                [
                    "P0",
                    "P50",
                    "PHIGH",
                ],

            "measured_bursts_per_cell":
                6,
        },
    }

    (
        out
        / "provenance.json"
    ).write_text(
        json.dumps(
            provenance,
            indent=2,
        )
        + "\n"
    )

    hashes = {}

    for name in [
        "all_executions.csv",
        "formal_bursts.csv",
        "formal_requests.csv",
        "provenance.json",
    ]:

        hashes[name] = (
            sha256_file(
                out / name
            )
        )

    (
        out
        / "artifact_hashes.json"
    ).write_text(
        json.dumps(
            hashes,
            indent=2,
        )
        + "\n"
    )

    print()
    print("=" * 80)
    print(
        "EXP11 FINAL MERGE: PASS"
    )
    print("=" * 80)

    print(
        "RESULT_DIR:",
        out,
    )

    print()

    for name, digest in (
        hashes.items()
    ):

        print(
            digest,
            name,
        )


if __name__ == "__main__":
    main()
