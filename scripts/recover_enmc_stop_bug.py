import copy
import json
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from src.comorag.ComoRAG import ComoRAG

from run_gemini_infinitebench import (
    PROJECT_ROOT,
    DATA_ROOT,
    SHARED_INDEX_ROOT,
    RESULT_ROOT,
    make_base_config,
    load_jsonl,
    get_index_marker_path,
    expected_index_metadata,
    index_marker_is_valid,
)


MANIFEST_PATH = (
    PROJECT_ROOT
    / "experiment_manifests"
    / "enmc_stop_bug_recovery.json"
)

RECOVERY_ROOT = (
    PROJECT_ROOT
    / "result"
    / "infinitebench"
    / "enmc_stop_bug_recovery"
)

FINAL_RECOVERY_FILE = (
    RECOVERY_ROOT
    / "results_recovered.json"
)


def save_json_atomic(path, data):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_name(
        path.name + ".tmp"
    )

    with temp.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    temp.replace(path)


def main():

    load_dotenv(
        PROJECT_ROOT / ".env"
    )

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found."
        )

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        manifest = json.load(f)

    if len(manifest) != 54:
        raise RuntimeError(
            "Recovery manifest should contain "
            f"54 questions, found {len(manifest)}."
        )

    by_context = defaultdict(list)

    for item in manifest:
        by_context[
            item["context_id"]
        ].append(
            item["idx"]
        )

    base_config = (
        make_base_config(
            api_key
        )
    )

    all_recovered = []

    for context_id in sorted(
        by_context
    ):

        original_indices = sorted(
            by_context[
                context_id
            ]
        )

        print()
        print("=" * 70)
        print(
            "Recovering context:",
            context_id
        )
        print(
            "Original indices:",
            original_indices
        )
        print("=" * 70)

        recovery_dir = (
            RECOVERY_ROOT
            / context_id
        )

        recovery_file = (
            recovery_dir
            / "results.json"
        )

        # ----------------------------------------------------
        # Resume support:
        # skip contexts already completely recovered.
        # ----------------------------------------------------

        if recovery_file.exists():

            with recovery_file.open(
                "r",
                encoding="utf-8",
            ) as f:
                existing = json.load(f)

            existing_indices = sorted(
                item["original_idx"]
                for item in existing
            )

            if (
                existing_indices
                == original_indices
            ):
                print(
                    "[SKIP] Recovery already complete."
                )

                all_recovered.extend(
                    existing
                )

                continue

            raise RuntimeError(
                "Existing recovery file has "
                "unexpected indices: "
                f"{recovery_file}"
            )

        # ----------------------------------------------------
        # Read the ORIGINAL bad-result file.
        # We use its exact question strings so the recovery
        # changes nothing except the corrupted model call.
        # ----------------------------------------------------

        source_results_path = (
            RESULT_ROOT
            / "enmc"
            / context_id
            / "results.json"
        )

        with source_results_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            source_results = json.load(f)

        source_by_idx = {
            item["idx"]: item
            for item in source_results
        }

        selected = []

        for idx in original_indices:

            if idx not in source_by_idx:
                raise RuntimeError(
                    f"Question idx {idx} missing "
                    f"from {source_results_path}"
                )

            selected.append(
                source_by_idx[idx]
            )

        final_queries = [
            item["question"]
            for item in selected
        ]

        retrieval_queries = [
            item["retrieval_question"]
            for item in selected
        ]

        # ----------------------------------------------------
        # Recreate EXACT experiment configuration,
        # but use the existing shared index.
        # ----------------------------------------------------

        config = copy.deepcopy(
            base_config
        )

        config.is_mc = True

        corpus_path = (
            DATA_ROOT
            / "enmc"
            / context_id
            / "corpus.jsonl"
        )

        corpus = load_jsonl(
            corpus_path
        )

        config.corpus_len = len(
            corpus
        )

        config.save_dir = str(
            SHARED_INDEX_ROOT
            / context_id
        )

        # Keep recovery details separate from the
        # original experiment files.
        config.output_dir = str(
            recovery_dir
        )

        config.dataset = None

        # ----------------------------------------------------
        # IMPORTANT:
        # Never rebuild an index during recovery.
        # ----------------------------------------------------

        marker_path = (
            get_index_marker_path(
                config
            )
        )

        expected_metadata = (
            expected_index_metadata(
                context_id=context_id,
                corpus_path=corpus_path,
                corpus_len=len(corpus),
                config=config,
            )
        )

        if not index_marker_is_valid(
            marker_path,
            expected_metadata,
        ):
            raise RuntimeError(
                "Valid completed index not found "
                f"for {context_id}. "
                "Recovery refuses to rebuild it."
            )

        print(
            "[INDEX] Reusing completed index."
        )

        # ----------------------------------------------------
        # Answer ONLY affected questions.
        # ----------------------------------------------------

        comorag = ComoRAG(
            global_config=config
        )

        solutions = (
            comorag.try_answer(
                final_queries,
                retrieval_queries=(
                    retrieval_queries
                ),
            )
        )

        if (
            len(solutions)
            != len(selected)
        ):
            raise RuntimeError(
                "Expected "
                f"{len(selected)} recovered "
                "solutions but received "
                f"{len(solutions)}."
            )

        recovered_rows = []

        for (
            local_idx,
            (
                original,
                solution,
            ),
        ) in enumerate(
            zip(
                selected,
                solutions,
            )
        ):

            recovered_rows.append(
                {
                    "context_id": (
                        context_id
                    ),
                    "original_idx": (
                        original["idx"]
                    ),
                    "recovery_local_idx": (
                        local_idx
                    ),
                    "question": (
                        original[
                            "question"
                        ]
                    ),
                    "retrieval_question": (
                        original[
                            "retrieval_question"
                        ]
                    ),
                    "golden_answers": (
                        original[
                            "golden_answers"
                        ]
                    ),
                    "old_output": (
                        original[
                            "output"
                        ]
                    ),
                    "new_output": (
                        solution.answer
                    ),
                }
            )

        save_json_atomic(
            recovery_file,
            recovered_rows,
        )

        all_recovered.extend(
            recovered_rows
        )

        print(
            "[DONE] Recovered",
            len(recovered_rows),
            "questions."
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    keys = [
        (
            item["context_id"],
            item["original_idx"],
        )
        for item in all_recovered
    ]

    if len(all_recovered) != 54:
        raise RuntimeError(
            "Expected 54 recovered questions, "
            f"found {len(all_recovered)}."
        )

    if len(set(keys)) != 54:
        raise RuntimeError(
            "Recovery contains duplicate "
            "question keys."
        )

    save_json_atomic(
        FINAL_RECOVERY_FILE,
        all_recovered,
    )

    no_output = [
        item
        for item in all_recovered
        if "<NO_OUTPUT>"
        in str(
            item["new_output"]
        )
    ]

    print()
    print("=" * 70)
    print("RECOVERY COMPLETE")
    print("=" * 70)

    print(
        "Recovered questions:",
        len(all_recovered)
    )

    print(
        "Recovered outputs currently "
        "containing <NO_OUTPUT>:",
        len(no_output)
    )

    print(
        "Saved:",
        FINAL_RECOVERY_FILE
    )


if __name__ == "__main__":
    main()