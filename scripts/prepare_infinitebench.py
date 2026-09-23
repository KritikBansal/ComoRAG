import hashlib
import json
import re
from pathlib import Path

from datasets import load_dataset, Features, Value, Sequence
from transformers import AutoTokenizer


features = Features({
    "id": Value("int64"),
    "context": Value("string"),
    "input": Value("string"),
    "answer": Sequence(Value("string")),
    "options": Sequence(Value("string")),
})

dataset = load_dataset(
    "xinrongzhang2022/InfiniteBench",
    features=features
)

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")


def gold_letter(answer, options):
    answer = answer.strip()

    match = re.match(
        r"^[\[\(\s]*([ABCD])[\]\)\.\s]*$",
        answer.upper()
    )

    if match:
        return match.group(1)

    normalized_answer = answer.strip().lower()

    for i, option in enumerate(options):
        if option.strip().lower() == normalized_answer:
            return chr(65 + i)

    raise ValueError(
        f"Cannot map answer '{answer}' to options {options}"
    )


def prepare(split_name, task_name, is_mc):
    rows = dataset[split_name]

    groups = {}

    for row in rows:
        context = row["context"]

        context_id = hashlib.sha256(
            context.encode("utf-8")
        ).hexdigest()[:16]

        if context_id not in groups:
            groups[context_id] = {
                "context": context,
                "questions": []
            }

        groups[context_id]["questions"].append(row)

    root = Path("dataset/infinitebench") / task_name

    for context_id, group in groups.items():
        folder = root / context_id
        folder.mkdir(parents=True, exist_ok=True)

        token_ids = tokenizer.encode(
            group["context"],
            add_special_tokens=False
        )

        chunks = [
            token_ids[i:i + 512]
            for i in range(0, len(token_ids), 512)
        ]

        with open(
            folder / "corpus.jsonl",
            "w",
            encoding="utf-8"
        ) as f:

            for i, chunk in enumerate(chunks):

                text = tokenizer.decode(
                    chunk,
                    skip_special_tokens=True
                )

                item = {
                    "id": i,
                    "doc_id": context_id,
                    "title": context_id,
                    "contents": text
                }

                f.write(json.dumps(
                    item,
                    ensure_ascii=False
                ) + "\n")

        with open(
            folder / "qas.jsonl",
            "w",
            encoding="utf-8"
        ) as f:

            for row in group["questions"]:

                retrieval_question = row["input"]

                if is_mc:
                    options = row["options"]

                    option_text = "\n".join(
                        f"{chr(65+i)}. {option}"
                        for i, option in enumerate(options)
                    )

                    full_question = (
                        retrieval_question
                        + "\nOptions:\n"
                        + option_text
                    )

                    golden_answers = [
                        gold_letter(
                            row["answer"][0],
                            options
                        )
                    ]

                else:
                    full_question = retrieval_question
                    golden_answers = list(row["answer"])

                item = {
                    "id": str(row["id"]),
                    "question": full_question,
                    "retrieval_question": retrieval_question,
                    "golden_answers": golden_answers
                }

                f.write(json.dumps(
                    item,
                    ensure_ascii=False
                ) + "\n")


prepare(
    "longbook_qa_eng",
    "enqa",
    False
)

prepare(
    "longbook_choice_eng",
    "enmc",
    True
)