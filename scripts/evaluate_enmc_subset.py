import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_PATH = (
    PROJECT_ROOT
    / "result"
    / "infinitebench"
    / "enmc"
    / "results_subset.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation_results"
    / "enmc_subset"
    / "evaluation_summary.json"
)

DETAIL_PATH = (
    PROJECT_ROOT
    / "evaluation_results"
    / "enmc_subset"
    / "detailed_evaluation_results.json"
)


def extract_final_answer(output_text):
    if not output_text:
        return ""

    marker = "### Final Answer"

    pos = output_text.rfind(marker)

    if pos == -1:
        return output_text.strip()

    return output_text[
        pos + len(marker):
    ].strip()


def official_infinitebench_score(
    pred,
    label,
):
    """
    Reproduce InfiniteBench's
    get_score_one_longbook_choice_eng().
    """

    pred = pred.strip()

    pattern = (
        r"\b[A-D]\b"
        r"(?!.*\b[A-D]\b)"
    )

    match = re.search(
        pattern,
        pred,
    )

    if match:
        extracted_pred = (
            match.group(0)
        )

        if extracted_pred in label:
            return True

    if pred == "":
        return False

    if pred[0] in "ABCD":
        return pred[0] in label

    if pred in label:
        return True

    for c in [
        "\n",
        '"',
        "'",
        ".",
        ",",
        "?",
        "!",
        "{",
        "}",
    ]:
        pred = pred.replace(
            c,
            " ",
        )

    while "  " in pred:
        pred = pred.replace(
            "  ",
            " ",
        )

    ans_prefixes = [
        "answer is:",
        "answer:",
        "answer is",
        "option is",
    ]

    for prefix in ans_prefixes:

        idx = pred.find(
            prefix
        )

        if idx == -1:
            continue

        if (
            len(pred)
            < idx
            + len(prefix)
            + 1
        ):
            return False

        after_prefix = pred[
            idx
            + len(prefix)
            + 1:
        ]

        for answer in label:

            if after_prefix.startswith(
                answer
            ):
                return True

        return False

    words = pred.split()

    for word in words:

        if word in "ABCD":
            return word in label

    return False


def official_extracted_letter(pred):
    """
    Diagnostic only:
    show which standalone option letter the
    main InfiniteBench regex sees.
    """

    pred = pred.strip()

    match = re.search(
        r"\b[A-D]\b"
        r"(?!.*\b[A-D]\b)",
        pred,
    )

    if match:
        return match.group(0)

    if (
        pred
        and pred[0] in "ABCD"
    ):
        return pred[0]

    return None


def strict_extract(pred):
    """
    Diagnostic metric.

    Accept only an explicit single-choice answer:
        A
        B
        C
        D
        [A]
        [B]
        [C]
        [D]

    Any prose/refusal counts as no valid choice.
    """

    pred = pred.strip()

    match = re.fullmatch(
        r"\[?\s*([A-D])\s*\]?",
        pred,
    )

    if match:
        return match.group(1)

    return None


def main():

    with RESULTS_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        results = json.load(f)

    if len(results) != 99:
        raise RuntimeError(
            "Expected 99 EN.MC answers, "
            f"found {len(results)}."
        )

    details = []

    official_correct = 0
    strict_correct = 0
    strict_valid = 0

    no_output_count = 0

    official_only_correct = []

    for number, item in enumerate(
        results
    ):

        prediction = (
            extract_final_answer(
                item.get(
                    "output",
                    "",
                )
            )
        )

        labels = item.get(
            "golden_answers",
            [],
        )

        if isinstance(
            labels,
            str,
        ):
            labels = [labels]

        official_correct_this = (
            official_infinitebench_score(
                prediction,
                labels,
            )
        )

        strict_letter = (
            strict_extract(
                prediction
            )
        )

        strict_valid_this = (
            strict_letter is not None
        )

        strict_correct_this = (
            strict_letter in labels
            if strict_letter
            is not None
            else False
        )

        official_letter = (
            official_extracted_letter(
                prediction
            )
        )

        if official_correct_this:
            official_correct += 1

        if strict_valid_this:
            strict_valid += 1

        if strict_correct_this:
            strict_correct += 1

        if (
            "<NO_OUTPUT>"
            in prediction
        ):
            no_output_count += 1

        if (
            official_correct_this
            and not strict_correct_this
        ):
            official_only_correct.append(
                number
            )

        details.append(
            {
                "number": number,
                "context_id": (
                    item.get(
                        "context_id"
                    )
                ),
                "idx": item.get(
                    "idx"
                ),
                "gold": labels,
                "prediction": prediction,
                "official_extracted_letter": (
                    official_letter
                ),
                "strict_extracted_letter": (
                    strict_letter
                ),
                "official_correct": (
                    official_correct_this
                ),
                "strict_correct": (
                    strict_correct_this
                ),
            }
        )

    total = len(results)

    summary = {
        "total": total,
        "official_correct": (
            official_correct
        ),
        "official_accuracy": (
            official_correct
            / total
        ),
        "strict_correct": (
            strict_correct
        ),
        "strict_accuracy": (
            strict_correct
            / total
        ),
        "strict_valid_choices": (
            strict_valid
        ),
        "strict_choice_rate": (
            strict_valid
            / total
        ),
        "no_output_count": (
            no_output_count
        ),
        "official_correct_but_not_strict": (
            len(
                official_only_correct
            )
        ),
        "official_only_indices": (
            official_only_correct
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    with DETAIL_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            details,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 65)
    print(
        "EN.MC EVALUATION"
    )
    print("=" * 65)

    print(
        "Total questions:",
        total,
    )

    print()

    print(
        "Official InfiniteBench:"
    )
    print(
        "  Correct:",
        official_correct,
    )
    print(
        "  Accuracy:",
        f"{100 * official_correct / total:.2f}%"
    )

    print()

    print(
        "Strict explicit-choice audit:"
    )
    print(
        "  Valid explicit choices:",
        strict_valid,
    )
    print(
        "  Choice-following rate:",
        f"{100 * strict_valid / total:.2f}%"
    )
    print(
        "  Correct:",
        strict_correct,
    )
    print(
        "  Accuracy:",
        f"{100 * strict_correct / total:.2f}%"
    )

    print()

    print(
        "<NO_OUTPUT>:",
        no_output_count,
    )

    print(
        "Official-correct but "
        "strict-incorrect:",
        len(
            official_only_correct
        ),
    )

    print()

    print(
        "Summary saved to:"
    )
    print(
        OUTPUT_PATH
    )

    print(
        "Detailed results saved to:"
    )
    print(
        DETAIL_PATH
    )


if __name__ == "__main__":
    main()