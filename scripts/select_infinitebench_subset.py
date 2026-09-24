import argparse
import csv
import hashlib
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "dataset" / "infinitebench"

ENQA_ROOT = DATA_ROOT / "enqa"
ENMC_ROOT = DATA_ROOT / "enmc"

INDEX_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "infinitebench"
    / "shared_indexes"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "result"
    / "infinitebench"
)

MANIFEST_ROOT = (
    PROJECT_ROOT
    / "experiment_manifests"
)


# ============================================================
# FILE HELPERS
# ============================================================

def count_jsonl(path: Path) -> int:
    """
    Count non-empty records in a JSONL file.
    """

    with path.open("r", encoding="utf-8") as f:
        return sum(
            1
            for line in f
            if line.strip()
        )


def sha256_file(path: Path) -> str:
    """
    Calculate SHA256 for a file.

    Used to verify that an EN.QA and EN.MC context with the
    same context ID really contains the same corpus.
    """

    sha = hashlib.sha256()

    with path.open("rb") as f:

        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            sha.update(block)

    return sha.hexdigest()


def nonempty_file(path: Path) -> bool:
    """
    Return True only if a file exists and contains data.
    """

    return (
        path.exists()
        and path.is_file()
        and path.stat().st_size > 0
    )


def git_commit():
    """
    Record the current Git commit for reproducibility.
    """

    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip()

    except Exception:
        return None


# ============================================================
# DATASET SCANNING
# ============================================================

def scan_task(task_root: Path):
    """
    Scan one prepared InfiniteBench task.

    Returns:
        {
            context_id: {
                "path": Path,
                "corpus_path": Path,
                "qas_path": Path,
                "chunks": int,
                "questions": int,
                "corpus_sha256": str,
            }
        }
    """

    contexts = {}

    if not task_root.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {task_root}"
        )

    for context_dir in sorted(task_root.iterdir()):

        if not context_dir.is_dir():
            continue

        context_id = context_dir.name

        corpus_path = (
            context_dir
            / "corpus.jsonl"
        )

        qas_path = (
            context_dir
            / "qas.jsonl"
        )

        if not corpus_path.exists():
            raise FileNotFoundError(
                f"Missing corpus.jsonl: {corpus_path}"
            )

        if not qas_path.exists():
            raise FileNotFoundError(
                f"Missing qas.jsonl: {qas_path}"
            )

        contexts[context_id] = {
            "path": context_dir,
            "corpus_path": corpus_path,
            "qas_path": qas_path,
            "chunks": count_jsonl(
                corpus_path
            ),
            "questions": count_jsonl(
                qas_path
            ),
            "corpus_sha256": sha256_file(
                corpus_path
            ),
        }

    return contexts


# ============================================================
# EXISTING WORK
# ============================================================

def has_completion_marker(
    context_id: str,
) -> bool:
    """
    Check whether this context already has a completed
    ComoRAG index marker.
    """

    context_root = (
        INDEX_ROOT
        / context_id
    )

    if not context_root.exists():
        return False

    markers = list(
        context_root.glob(
            "*/.index_complete.json"
        )
    )

    return any(
        nonempty_file(marker)
        for marker in markers
    )


def has_task_result(
    task_name: str,
    context_id: str,
) -> bool:
    """
    Check whether this context already has a saved task result.
    """

    result_file = (
        RESULT_ROOT
        / task_name
        / context_id
        / "results.json"
    )

    return nonempty_file(
        result_file
    )


# ============================================================
# CONTEXT HELPERS
# ============================================================

def get_context_chunks(
    context_id: str,
    enqa: dict,
    enmc: dict,
) -> int:
    """
    Get the chunk count for a context.

    For shared contexts EN.QA and EN.MC are first verified
    to contain identical corpora.
    """

    if context_id in enqa:
        return enqa[context_id]["chunks"]

    return enmc[context_id]["chunks"]


def get_question_count(
    context_id: str,
    task_data: dict,
) -> int:
    """
    Number of questions for one context in one task.
    """

    if context_id not in task_data:
        return 0

    return task_data[
        context_id
    ]["questions"]


# ============================================================
# VERIFY SHARED CONTEXTS
# ============================================================

def verify_shared_contexts(
    enqa: dict,
    enmc: dict,
):
    """
    Contexts with the same ID are supposed to represent
    the same long document.

    Since we intend to reuse the same ComoRAG index across
    EN.QA and EN.MC, verify the corpus contents really match.
    """

    shared_ids = (
        set(enqa)
        & set(enmc)
    )

    for context_id in shared_ids:

        qa_hash = (
            enqa[context_id]
            ["corpus_sha256"]
        )

        mc_hash = (
            enmc[context_id]
            ["corpus_sha256"]
        )

        if qa_hash != mc_hash:

            raise RuntimeError(
                "\nShared-index safety check failed.\n"
                f"Context: {context_id}\n"
                "EN.QA and EN.MC corpus.jsonl files differ.\n"
                "Do not share an index for this context."
            )

    return shared_ids


# ============================================================
# SIZE STRATIFICATION
# ============================================================

def make_size_bands(
    context_ids,
    enqa: dict,
    enmc: dict,
):
    """
    Divide contexts into approximately equal thirds based
    on chunk count.

    The bands are rank-based rather than using arbitrary
    fixed cutoffs.

        small
        medium
        large
    """

    ordered = sorted(
        context_ids,
        key=lambda cid: (
            get_context_chunks(
                cid,
                enqa,
                enmc,
            ),
            cid,
        )
    )

    n = len(ordered)

    bands = {
        "small": [],
        "medium": [],
        "large": [],
    }

    band_for_context = {}

    if n == 0:
        return (
            bands,
            band_for_context,
        )

    for rank, context_id in enumerate(
        ordered
    ):

        # Produces approximately equal thirds.
        band_number = min(
            2,
            (3 * rank) // n
        )

        band_name = (
            "small"
            if band_number == 0
            else
            "medium"
            if band_number == 1
            else
            "large"
        )

        bands[
            band_name
        ].append(
            context_id
        )

        band_for_context[
            context_id
        ] = band_name

    return (
        bands,
        band_for_context,
    )


# ============================================================
# MANIFEST WRITING
# ============================================================

def write_json(
    path: Path,
    data,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def write_context_list(
    path: Path,
    context_ids,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for context_id in context_ids:
            f.write(
                context_id
                + "\n"
            )


def write_csv(
    path: Path,
    records,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "context_id",
        "shared",
        "size_band",
        "chunks",
        "enqa_questions",
        "enmc_questions",
        "already_indexed",
        "enqa_result_exists",
        "enmc_result_exists",
        "already_paid",
        "selection_reason",
        "estimated_incremental_index_cost_eur",
    ]

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

        for record in records:

            writer.writerow(
                {
                    key: record.get(
                        key
                    )
                    for key in fields
                }
            )


# ============================================================
# MAIN SELECTION
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create a budget-capped, stratified "
            "InfiniteBench EN.QA/EN.MC subset."
        )
    )

    parser.add_argument(
        "--budget-cap-eur",
        type=float,
        default=30.0,
        help=(
            "Hard total API budget. "
            "Default: 30 EUR."
        ),
    )

    parser.add_argument(
        "--already-spent-eur",
        type=float,
        default=5.0,
        help=(
            "Approximate Gemini API spending "
            "already incurred."
        ),
    )

    parser.add_argument(
        "--reserve-eur",
        type=float,
        default=7.0,
        help=(
            "Budget reserved for QA/MC answering, "
            "retries and API variability. "
            "Default: 7 EUR."
        ),
    )

    parser.add_argument(
        "--observed-cost-eur",
        type=float,
        default=5.0,
        help=(
            "Observed API cost used for empirical "
            "cost calibration."
        ),
    )

    parser.add_argument(
        "--observed-chunks",
        type=int,
        default=2413,
        help=(
            "Number of chunks corresponding to "
            "the observed API cost."
        ),
    )

    parser.add_argument(
        "--observed-hours",
        type=float,
        default=4.0,
        help=(
            "Observed runtime corresponding to "
            "the observed chunks."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Fixed random seed for reproducible "
            "stratified selection."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Optional output JSON path. "
            "Defaults to experiment_manifests/"
            "infinitebench_subset_seed<seed>.json"
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate budget inputs
    # --------------------------------------------------------

    if args.budget_cap_eur <= 0:
        raise ValueError(
            "budget-cap-eur must be > 0"
        )

    if args.observed_cost_eur <= 0:
        raise ValueError(
            "observed-cost-eur must be > 0"
        )

    if args.observed_chunks <= 0:
        raise ValueError(
            "observed-chunks must be > 0"
        )

    if args.observed_hours <= 0:
        raise ValueError(
            "observed-hours must be > 0"
        )

    if args.already_spent_eur < 0:
        raise ValueError(
            "already-spent-eur cannot be negative"
        )

    if args.reserve_eur < 0:
        raise ValueError(
            "reserve-eur cannot be negative"
        )

    available_new_index_budget = (
        args.budget_cap_eur
        - args.already_spent_eur
        - args.reserve_eur
    )

    if available_new_index_budget < 0:

        raise ValueError(
            "Budget cap is smaller than "
            "already-spent + reserve."
        )

    # --------------------------------------------------------
    # Empirical rates
    # --------------------------------------------------------

    cost_per_chunk = (
        args.observed_cost_eur
        / args.observed_chunks
    )

    hours_per_chunk = (
        args.observed_hours
        / args.observed_chunks
    )

    # --------------------------------------------------------
    # Scan datasets
    # --------------------------------------------------------

    print(
        "Scanning InfiniteBench..."
    )

    enqa = scan_task(
        ENQA_ROOT
    )

    enmc = scan_task(
        ENMC_ROOT
    )

    shared_ids = verify_shared_contexts(
        enqa,
        enmc,
    )

    all_ids = (
        set(enqa)
        | set(enmc)
    )

    qa_only_ids = (
        set(enqa)
        - set(enmc)
    )

    mc_only_ids = (
        set(enmc)
        - set(enqa)
    )

    # --------------------------------------------------------
    # Existing work
    #
    # Existing indexes/results are always retained because
    # their cost has already been incurred.
    #
    # We do NOT inspect prediction contents or scores.
    # --------------------------------------------------------

    context_status = {}

    mandatory_ids = set()

    for context_id in all_ids:

        indexed = has_completion_marker(
            context_id
        )

        qa_result = has_task_result(
            "enqa",
            context_id,
        )

        mc_result = has_task_result(
            "enmc",
            context_id,
        )

        already_paid = (
            indexed
            or qa_result
            or mc_result
        )

        context_status[
            context_id
        ] = {
            "already_indexed": indexed,
            "enqa_result_exists": qa_result,
            "enmc_result_exists": mc_result,
            "already_paid": already_paid,
        }

        if already_paid:
            mandatory_ids.add(
                context_id
            )

    # --------------------------------------------------------
    # Stratify SHARED contexts.
    #
    # New selection is deliberately restricted to contexts
    # available in BOTH EN.QA and EN.MC.
    #
    # This maximizes index reuse and gives both tasks a common
    # core evaluation subset.
    # --------------------------------------------------------

    (
        size_bands,
        band_for_context,
    ) = make_size_bands(
        shared_ids,
        enqa,
        enmc,
    )

    # --------------------------------------------------------
    # Reproducibly randomize within each size stratum.
    # --------------------------------------------------------

    shuffled_bands = {}

    seed_offsets = {
        "small": 101,
        "medium": 202,
        "large": 303,
    }

    for band_name in [
        "small",
        "medium",
        "large",
    ]:

        candidates = [
            cid
            for cid
            in size_bands[
                band_name
            ]
            if cid
            not in mandatory_ids
        ]

        rng = random.Random(
            args.seed
            + seed_offsets[
                band_name
            ]
        )

        rng.shuffle(
            candidates
        )

        shuffled_bands[
            band_name
        ] = candidates

    # --------------------------------------------------------
    # Start with already-paid contexts.
    # --------------------------------------------------------

    selected_ids = set(
        mandatory_ids
    )

    selection_reason = {
        cid: "existing_artifact"
        for cid
        in mandatory_ids
    }

    incremental_cost = 0.0
    incremental_chunks = 0

    # --------------------------------------------------------
    # Round-robin selection:
    #
    # small -> medium -> large -> small -> ...
    #
    # This prevents the subset from containing only short
    # cheap books.
    # --------------------------------------------------------

    band_order = [
        "small",
        "medium",
        "large",
    ]

    positions = {
        name: 0
        for name
        in band_order
    }

    while True:

        found_candidate = False
        added_candidate = False

        for band_name in band_order:

            candidates = (
                shuffled_bands[
                    band_name
                ]
            )

            while (
                positions[
                    band_name
                ]
                < len(candidates)
            ):

                context_id = candidates[
                    positions[
                        band_name
                    ]
                ]

                positions[
                    band_name
                ] += 1

                found_candidate = True

                chunks = get_context_chunks(
                    context_id,
                    enqa,
                    enmc,
                )

                estimated_cost = (
                    chunks
                    * cost_per_chunk
                )

                new_total_cost = (
                    incremental_cost
                    + estimated_cost
                )

                # Candidate fits within the portion of
                # the budget available for new indexing.
                if (
                    new_total_cost
                    <= available_new_index_budget
                    + 1e-9
                ):

                    selected_ids.add(
                        context_id
                    )

                    selection_reason[
                        context_id
                    ] = (
                        "budget_stratified_shared"
                    )

                    incremental_cost += (
                        estimated_cost
                    )

                    incremental_chunks += (
                        chunks
                    )

                    added_candidate = True

                    break

                # If it does not fit, permanently skip it.
                # Continue looking for a smaller candidate
                # in the same stratum.

        if not found_candidate:
            break

        if not added_candidate:
            break

    # --------------------------------------------------------
    # Construct task-specific selections.
    # --------------------------------------------------------

    selected_enqa = sorted(
        cid
        for cid in selected_ids
        if cid in enqa
    )

    selected_enmc = sorted(
        cid
        for cid in selected_ids
        if cid in enmc
    )

    selected_shared = sorted(
        cid
        for cid in selected_ids
        if cid in shared_ids
    )

    # --------------------------------------------------------
    # Context records
    # --------------------------------------------------------

    records = []

    total_selected_unique_chunks = 0

    selected_enqa_questions = 0
    selected_enmc_questions = 0

    for context_id in sorted(
        selected_ids
    ):

        chunks = get_context_chunks(
            context_id,
            enqa,
            enmc,
        )

        total_selected_unique_chunks += (
            chunks
        )

        qa_questions = (
            get_question_count(
                context_id,
                enqa,
            )
        )

        mc_questions = (
            get_question_count(
                context_id,
                enmc,
            )
        )

        selected_enqa_questions += (
            qa_questions
        )

        selected_enmc_questions += (
            mc_questions
        )

        status = context_status[
            context_id
        ]

        if status[
            "already_paid"
        ]:

            estimated_incremental_cost = (
                0.0
            )

        else:

            estimated_incremental_cost = (
                chunks
                * cost_per_chunk
            )

        if context_id in shared_ids:

            size_band = (
                band_for_context[
                    context_id
                ]
            )

        else:
            size_band = (
                "task_only_existing"
            )

        records.append(
            {
                "context_id": context_id,

                "shared": (
                    context_id
                    in shared_ids
                ),

                "size_band": size_band,

                "chunks": chunks,

                "enqa_questions": (
                    qa_questions
                ),

                "enmc_questions": (
                    mc_questions
                ),

                "already_indexed": status[
                    "already_indexed"
                ],

                "enqa_result_exists": status[
                    "enqa_result_exists"
                ],

                "enmc_result_exists": status[
                    "enmc_result_exists"
                ],

                "already_paid": status[
                    "already_paid"
                ],

                "selection_reason": (
                    selection_reason[
                        context_id
                    ]
                ),

                "estimated_incremental_index_cost_eur": round(
                    estimated_incremental_cost,
                    4,
                ),
            }
        )

    # --------------------------------------------------------
    # Cost / runtime projection
    # --------------------------------------------------------

    projected_spend_before_reserve = (
        args.already_spent_eur
        + incremental_cost
    )

    projected_budget_with_reserve = (
        projected_spend_before_reserve
        + args.reserve_eur
    )

    estimated_new_index_hours = (
        incremental_chunks
        * hours_per_chunk
    )

    # --------------------------------------------------------
    # Dataset totals
    # --------------------------------------------------------

    total_enqa_questions = sum(
        item["questions"]
        for item in enqa.values()
    )

    total_enmc_questions = sum(
        item["questions"]
        for item in enmc.values()
    )

    total_enqa_chunks = sum(
        item["chunks"]
        for item in enqa.values()
    )

    # Unique corpus size:
    # all EN.QA contexts plus MC-only contexts.
    total_unique_chunks = (
        total_enqa_chunks
        + sum(
            enmc[cid]["chunks"]
            for cid
            in mc_only_ids
        )
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = {
        "schema_version": 1,

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "git_commit": git_commit(),

        "selection_method": (
            "budget-capped stratified selection over shared "
            "EN.QA/EN.MC contexts; existing paid contexts are "
            "retained; no model scores are inspected"
        ),

        "parameters": {
            "budget_cap_eur": (
                args.budget_cap_eur
            ),

            "already_spent_eur": (
                args.already_spent_eur
            ),

            "reserve_eur": (
                args.reserve_eur
            ),

            "seed": (
                args.seed
            ),

            "observed_cost_eur": (
                args.observed_cost_eur
            ),

            "observed_chunks": (
                args.observed_chunks
            ),

            "observed_hours": (
                args.observed_hours
            ),

            "estimated_cost_per_chunk_eur": round(
                cost_per_chunk,
                8,
            ),

            "estimated_hours_per_chunk": round(
                hours_per_chunk,
                8,
            ),
        },

        "full_dataset": {
            "enqa_contexts": len(
                enqa
            ),

            "enmc_contexts": len(
                enmc
            ),

            "shared_contexts": len(
                shared_ids
            ),

            "enqa_only_contexts": len(
                qa_only_ids
            ),

            "enmc_only_contexts": len(
                mc_only_ids
            ),

            "unique_contexts": len(
                all_ids
            ),

            "enqa_questions": (
                total_enqa_questions
            ),

            "enmc_questions": (
                total_enmc_questions
            ),

            "unique_chunks": (
                total_unique_chunks
            ),
        },

        "selected_subset": {
            "unique_contexts": len(
                selected_ids
            ),

            "shared_contexts": len(
                selected_shared
            ),

            "enqa_contexts": len(
                selected_enqa
            ),

            "enmc_contexts": len(
                selected_enmc
            ),

            "unique_chunks": (
                total_selected_unique_chunks
            ),

            "enqa_questions": (
                selected_enqa_questions
            ),

            "enmc_questions": (
                selected_enmc_questions
            ),

            "selected_shared_context_ids": (
                selected_shared
            ),

            "selected_enqa_context_ids": (
                selected_enqa
            ),

            "selected_enmc_context_ids": (
                selected_enmc
            ),
        },

        "budget_projection": {
            "available_for_new_indexing_eur": round(
                available_new_index_budget,
                2,
            ),

            "new_index_chunks": (
                incremental_chunks
            ),

            "estimated_new_index_cost_eur": round(
                incremental_cost,
                2,
            ),

            "estimated_new_index_hours": round(
                estimated_new_index_hours,
                2,
            ),

            "projected_spend_before_reserve_eur": round(
                projected_spend_before_reserve,
                2,
            ),

            "reserved_for_answering_retries_eur": round(
                args.reserve_eur,
                2,
            ),

            "projected_total_with_reserve_eur": round(
                projected_budget_with_reserve,
                2,
            ),

            "hard_budget_cap_eur": round(
                args.budget_cap_eur,
                2,
            ),
        },

        "contexts": records,
    }

    # --------------------------------------------------------
    # Output filenames
    # --------------------------------------------------------

    if args.output is None:

        output_path = (
            MANIFEST_ROOT
            / (
                "infinitebench_subset_"
                f"seed{args.seed}.json"
            )
        )

    else:

        output_path = Path(
            args.output
        )

        if not output_path.is_absolute():

            output_path = (
                PROJECT_ROOT
                / output_path
            )

    csv_path = (
        output_path
        .with_suffix(".csv")
    )

    enqa_list_path = (
        output_path.parent
        / (
            output_path.stem
            + "_enqa.txt"
        )
    )

    enmc_list_path = (
        output_path.parent
        / (
            output_path.stem
            + "_enmc.txt"
        )
    )

    # --------------------------------------------------------
    # Write outputs
    # --------------------------------------------------------

    write_json(
        output_path,
        manifest,
    )

    write_csv(
        csv_path,
        records,
    )

    write_context_list(
        enqa_list_path,
        selected_enqa,
    )

    write_context_list(
        enmc_list_path,
        selected_enmc,
    )

    # --------------------------------------------------------
    # Human-readable summary
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("INFINITEBENCH SUBSET SELECTION")
    print("=" * 72)

    print()
    print("Full dataset:")
    print(
        f"  EN.QA contexts:       {len(enqa)}"
    )
    print(
        f"  EN.MC contexts:       {len(enmc)}"
    )
    print(
        f"  Shared contexts:      {len(shared_ids)}"
    )
    print(
        f"  Unique contexts:      {len(all_ids)}"
    )

    print()
    print("Already-paid contexts:")
    print(
        f"  {len(mandatory_ids)}"
    )

    print()
    print("Selected subset:")
    print(
        f"  Unique contexts:      {len(selected_ids)}"
    )
    print(
        f"  Shared contexts:      {len(selected_shared)}"
    )
    print(
        f"  EN.QA contexts:       {len(selected_enqa)}"
    )
    print(
        f"  EN.MC contexts:       {len(selected_enmc)}"
    )
    print(
        f"  Unique chunks:        {total_selected_unique_chunks}"
    )
    print(
        f"  EN.QA questions:      {selected_enqa_questions}"
    )
    print(
        f"  EN.MC questions:      {selected_enmc_questions}"
    )

    print()
    print("Empirical rate:")
    print(
        f"  Cost/chunk:           €{cost_per_chunk:.6f}"
    )
    print(
        f"  Seconds/chunk:        {hours_per_chunk * 3600:.2f}"
    )

    print()
    print("Budget:")
    print(
        f"  Hard cap:             €{args.budget_cap_eur:.2f}"
    )
    print(
        f"  Already spent:        €{args.already_spent_eur:.2f}"
    )
    print(
        f"  Safety reserve:       €{args.reserve_eur:.2f}"
    )
    print(
        f"  New indexing budget:  €{available_new_index_budget:.2f}"
    )
    print(
        f"  Est. new index cost:  €{incremental_cost:.2f}"
    )
    print(
        f"  Est. total + reserve: €{projected_budget_with_reserve:.2f}"
    )

    print()
    print("Runtime:")
    print(
        f"  Est. new indexing:    {estimated_new_index_hours:.2f} hours"
    )

    print()
    print("Files written:")
    print(
        f"  Manifest: {output_path}"
    )
    print(
        f"  CSV:      {csv_path}"
    )
    print(
        f"  EN.QA:    {enqa_list_path}"
    )
    print(
        f"  EN.MC:    {enmc_list_path}"
    )

    print()
    print(
        "IMPORTANT: freeze these files before running "
        "the subset experiment."
    )


if __name__ == "__main__":
    main()