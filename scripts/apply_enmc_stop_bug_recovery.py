import json
import shutil
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

ENMC_ROOT = (
    PROJECT_ROOT
    / "result"
    / "infinitebench"
    / "enmc"
)

RECOVERY_FILE = (
    PROJECT_ROOT
    / "result"
    / "infinitebench"
    / "enmc_stop_bug_recovery"
    / "results_recovered.json"
)

BACKUP_ROOT = (
    PROJECT_ROOT
    / "experiment_manifests"
    / "enmc_pre_stop_bug_recovery"
)


def save_json_atomic(
    path,
    data,
):

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

    recovery = json.load(
        RECOVERY_FILE.open(
            encoding="utf-8"
        )
    )

    if len(recovery) != 54:
        raise RuntimeError(
            "Expected exactly 54 "
            "recovered answers."
        )

    replacements = {
        (
            item["context_id"],
            item["original_idx"],
        ): item["new_output"]
        for item in recovery
    }

    if len(replacements) != 54:
        raise RuntimeError(
            "Duplicate recovery keys."
        )

    BACKUP_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------
    # Backup aggregate before editing
    # ----------------------------------------------------

    aggregate_path = (
        ENMC_ROOT
        / "results_subset.json"
    )

    aggregate_backup = (
        BACKUP_ROOT
        / "results_subset.json"
    )

    if not aggregate_backup.exists():

        shutil.copy2(
            aggregate_path,
            aggregate_backup,
        )

    # ----------------------------------------------------
    # Update affected context-level result files
    # ----------------------------------------------------

    contexts = sorted(
        {
            context_id
            for context_id, _
            in replacements
        }
    )

    total_context_replacements = 0

    for context_id in contexts:

        result_path = (
            ENMC_ROOT
            / context_id
            / "results.json"
        )

        backup_dir = (
            BACKUP_ROOT
            / context_id
        )

        backup_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        backup_path = (
            backup_dir
            / "results.json"
        )

        if not backup_path.exists():

            shutil.copy2(
                result_path,
                backup_path,
            )

        data = json.load(
            result_path.open(
                encoding="utf-8"
            )
        )

        changed = 0

        for item in data:

            key = (
                context_id,
                item["idx"],
            )

            if key in replacements:

                item["output"] = (
                    replacements[key]
                )

                changed += 1

        expected_here = sum(
            1
            for key
            in replacements
            if key[0] == context_id
        )

        if changed != expected_here:
            raise RuntimeError(
                f"{context_id}: expected "
                f"{expected_here} replacements, "
                f"made {changed}."
            )

        save_json_atomic(
            result_path,
            data,
        )

        total_context_replacements += (
            changed
        )

    # ----------------------------------------------------
    # Update frozen aggregate
    # ----------------------------------------------------

    aggregate = json.load(
        aggregate_path.open(
            encoding="utf-8"
        )
    )

    if len(aggregate) != 99:
        raise RuntimeError(
            "Expected 99 aggregate "
            "EN.MC results."
        )

    aggregate_changed = 0

    for item in aggregate:

        key = (
            item["context_id"],
            item["idx"],
        )

        if key in replacements:

            item["output"] = (
                replacements[key]
            )

            aggregate_changed += 1

    if aggregate_changed != 54:
        raise RuntimeError(
            "Expected to replace 54 "
            "aggregate outputs, replaced "
            f"{aggregate_changed}."
        )

    save_json_atomic(
        aggregate_path,
        aggregate,
    )

    print("=" * 70)
    print(
        "EN.MC RECOVERY APPLIED"
    )
    print("=" * 70)

    print(
        "Context-level replacements:",
        total_context_replacements,
    )

    print(
        "Aggregate replacements:",
        aggregate_changed,
    )

    print(
        "Backup:",
        BACKUP_ROOT,
    )


if __name__ == "__main__":
    main()