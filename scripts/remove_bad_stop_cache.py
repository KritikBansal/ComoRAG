import json
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CACHE_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "infinitebench"
    / "shared_indexes"
)

AUDIT_PATH = (
    PROJECT_ROOT
    / "experiment_manifests"
    / "removed_bad_stop_cache_rows.json"
)


def main():

    removed = []

    db_files = list(
        CACHE_ROOT.glob(
            "*/llm_cache/*.sqlite"
        )
    )

    for db_path in db_files:

        conn = sqlite3.connect(
            db_path
        )

        rows = conn.execute(
            """
            SELECT key, message, metadata
            FROM cache
            """
        ).fetchall()

        keys_to_delete = []

        for (
            key,
            message,
            metadata_text,
        ) in rows:

            if (
                "<NO_OUTPUT>"
                not in str(message)
            ):
                continue

            try:
                metadata = json.loads(
                    metadata_text
                )
            except Exception:
                continue

            finish_reason = str(
                metadata.get(
                    "finish_reason",
                    "",
                )
            ).strip().lower()

            blocked = metadata.get(
                "blocked",
                False,
            )

            # Only remove the erroneous responses
            # produced by our indentation bug.
            #
            # Genuine PROHIBITED_CONTENT entries stay.
            if (
                finish_reason == "stop"
                and blocked is True
            ):

                keys_to_delete.append(
                    key
                )

                removed.append(
                    {
                        "database": str(
                            db_path.relative_to(
                                PROJECT_ROOT
                            )
                        ),
                        "key": key,
                        "message": message,
                        "metadata": metadata,
                    }
                )

        for key in keys_to_delete:

            conn.execute(
                """
                DELETE FROM cache
                WHERE key = ?
                """,
                (key,),
            )

        conn.commit()
        conn.close()

    AUDIT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with AUDIT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            removed,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 70)
    print("BAD STOP/<NO_OUTPUT> CACHE CLEANUP")
    print("=" * 70)

    print(
        "Removed cache rows:",
        len(removed),
    )

    print(
        "Audit saved to:",
        AUDIT_PATH,
    )

    print()
    print(
        "Genuine content-filter cache rows "
        "were NOT removed."
    )


if __name__ == "__main__":
    main()