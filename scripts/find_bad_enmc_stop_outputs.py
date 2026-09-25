import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_ROOT = (
    PROJECT_ROOT
    / "result"
    / "infinitebench"
    / "enmc"
)

AGGREGATE_PATH = (
    RESULT_ROOT
    / "results_subset.json"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "experiment_manifests"
    / "enmc_stop_bug_recovery.json"
)


def main():

    with AGGREGATE_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        results = json.load(f)

    affected = []

    for item in results:

        output = str(
            item.get(
                "output",
                "",
            )
        )

        if (
            "<NO_OUTPUT>"
            not in output
        ):
            continue

        context_id = item[
            "context_id"
        ]

        idx = item["idx"]

        detail_path = (
            RESULT_ROOT
            / context_id
            / "details"
            / f"qa_output_{idx}.txt"
        )

        if not detail_path.exists():
            raise RuntimeError(
                "Missing detail file: "
                f"{detail_path}"
            )

        detail = (
            detail_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            "Gemini generation blocked: stop"
            in detail
        ):

            affected.append(
                {
                    "context_id": (
                        context_id
                    ),
                    "idx": idx,
                }
            )

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            affected,
            f,
            indent=2,
        )

    contexts = sorted(
        {
            x["context_id"]
            for x in affected
        }
    )

    print("=" * 65)
    print(
        "EN.MC STOP-BUG RECOVERY MANIFEST"
    )
    print("=" * 65)

    print(
        "Affected questions:",
        len(affected),
    )

    print(
        "Affected contexts:",
        len(contexts),
    )

    print()

    for context in contexts:

        indices = [
            x["idx"]
            for x in affected
            if x["context_id"]
            == context
        ]

        print(
            context,
            indices,
        )

    print()
    print(
        "Saved:",
        MANIFEST_PATH,
    )


if __name__ == "__main__":
    main()