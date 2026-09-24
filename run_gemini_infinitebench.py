import argparse
import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from src.comorag.ComoRAG import ComoRAG
from src.comorag.utils.config_utils import BaseConfig
from src.comorag.utils.misc_utils import get_gold_answers


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_ROOT = PROJECT_ROOT / "dataset" / "infinitebench"

# IMPORTANT:
# EN.QA and EN.MC use the SAME index directory when their
# context_hash is identical.
SHARED_INDEX_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "infinitebench"
    / "shared_indexes"
)

# QA results are NOT shared.
RESULT_ROOT = (
    PROJECT_ROOT
    / "result"
    / "infinitebench"
)


# ============================================================
# MODEL / EXPERIMENT SETTINGS
# ============================================================

MODEL_NAME = "gemini-3.1-flash-lite"

GEMINI_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai/"
)

EMBEDDING_MODEL = "BAAI/bge-m3"


# ============================================================
# JSONL HELPERS
# ============================================================

def load_jsonl(path: Path):
    """Read a JSONL file into a list of dictionaries."""

    with path.open("r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def save_json(path: Path, data):
    """Write pretty-formatted JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

# ============================================================
# INDEX COMPLETION / RESUME HELPERS
# ============================================================

def sha256_file(path: Path) -> str:
    """
    Calculate the SHA256 fingerprint of a file.

    We use this to make sure an existing shared index was
    actually built from the same corpus.jsonl.
    """

    sha256 = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            sha256.update(block)

    return sha256.hexdigest()


def get_model_working_dir(config: BaseConfig) -> Path:
    """
    Reproduce ComoRAG's working-directory naming scheme.

    ComoRAG internally creates:

        <save_dir>/<llm_name>_<embedding_model_name>/

    with "/" replaced by "_".
    """

    llm_label = config.llm_name.replace("/", "_")
    embedding_label = config.embedding_model_name.replace("/", "_")

    return (
        Path(config.save_dir)
        / f"{llm_label}_{embedding_label}"
    )


def get_index_marker_path(config: BaseConfig) -> Path:
    """
    Location of the successful-index marker.
    """

    return (
        get_model_working_dir(config)
        / ".index_complete.json"
    )


def expected_index_metadata(
    context_id: str,
    corpus_path: Path,
    corpus_len: int,
    config: BaseConfig,
):
    """
    Information that must match before an existing index
    may be reused.
    """

    return {
        "context_id": context_id,
        "corpus_sha256": sha256_file(corpus_path),
        "corpus_len": corpus_len,

        "llm_name": config.llm_name,
        "embedding_model_name": config.embedding_model_name,

        "openie_mode": config.openie_mode,
        "need_cluster": config.need_cluster,

        "temperature": config.temperature,
        "seed": config.seed,
    }


def index_marker_is_valid(
    marker_path: Path,
    expected_metadata: dict,
) -> bool:
    """
    Return True only if:

    1. the marker exists;
    2. it contains valid JSON;
    3. all relevant settings match the current run.
    """

    if not marker_path.exists():
        return False

    try:
        with marker_path.open(
            "r",
            encoding="utf-8"
        ) as f:
            marker = json.load(f)

    except (json.JSONDecodeError, OSError):
        print(
            f"[INDEX] Invalid marker file: {marker_path}"
        )
        return False

    for key, expected_value in expected_metadata.items():

        actual_value = marker.get(key)

        if actual_value != expected_value:
            print(
                f"[INDEX] Marker mismatch for '{key}': "
                f"stored={actual_value!r}, "
                f"current={expected_value!r}"
            )
            return False

    return True

def verify_index_artifacts(config: BaseConfig):
    """
    Verify that the important ComoRAG index artifacts exist
    before declaring the index complete.

    Returns:
        (True, []) if everything looks complete.

        (False, missing_files) otherwise.
    """

    working_dir = get_model_working_dir(config)

    required_files = [
        working_dir
        / "chunk_embeddings"
        / "vdb_chunk.parquet",

        working_dir
        / "entity_embeddings"
        / "vdb_entity.parquet",

        working_dir
        / "fact_embeddings"
        / "vdb_fact.parquet",

        working_dir
        / "summary_embeddings"
        / "vdb_summary.parquet",

        working_dir
        / "timeline_embeddings"
        / "vdb_level_0.parquet",

        working_dir
        / "graph.graphml",

        working_dir
        / "final_summary.txt",
    ]

    missing_files = [
        path
        for path in required_files
        if not path.exists() or path.stat().st_size == 0
    ]

    return len(missing_files) == 0, missing_files

def write_index_marker(
    marker_path: Path,
    metadata: dict,
):
    """
    Atomically create the index-completion marker.

    The temporary file is written first. Only after the JSON
    has been completely written do we rename it to the real
    marker.
    """

    marker_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    marker_data = {
        **metadata,

        "status": "complete",

        "completed_at_utc": (
            datetime.now(timezone.utc)
            .isoformat()
        ),
    }

    temp_marker = marker_path.with_suffix(
        ".json.tmp"
    )

    with temp_marker.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            marker_data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    temp_marker.replace(marker_path)

    print(
        f"[INDEX] Completion marker written:"
    )
    print(
        f"        {marker_path}"
    )

# ============================================================
# CONFIGURATION
# ============================================================

def make_base_config(api_key: str):
    """
    Create settings shared by EN.QA and EN.MC.

    Task-specific fields such as is_mc, save_dir and
    output_dir are assigned later for each context.
    """

    config = BaseConfig(
        # ----------------------------
        # Gemini
        # ----------------------------
        llm_name=MODEL_NAME,
        llm_base_url=GEMINI_BASE_URL,
        llm_api_key=api_key,

        # Use API-based OpenIE, not vLLM
        openie_mode="online",

        # ----------------------------
        # Embeddings
        # ----------------------------
        embedding_model_name=EMBEDDING_MODEL,
        embedding_batch_size=32,

        # ----------------------------
        # ComoRAG structure
        # ----------------------------
        need_cluster=True,

        # ----------------------------
        # Reproducibility
        # ----------------------------
        seed=0,
        temperature=0,

        # ----------------------------
        # Cognitive loop
        # ----------------------------
        max_meta_loop_max_iterations=5,

        # ----------------------------
        # Evidence budget
        #
        # Our reconstruction of the
        # paper's 6000-token 8:2:2:1
        # V:S:E:H allocation.
        # ----------------------------
        max_tokens_ver=3692,
        max_tokens_sem=923,
        max_tokens_epi=923,

        # Default here. We overwrite
        # it per task below.
        is_mc=False,
    )

    # We added this field in our earlier ComoRAG patch.
    # Keeping it outside BaseConfig(...) makes this runner
    # slightly easier to diagnose if the field was not added.
    if not hasattr(config, "max_tokens_hist"):
        raise RuntimeError(
            "BaseConfig has no max_tokens_hist field. "
            "Apply the earlier historical-memory budget patch first."
        )

    config.max_tokens_hist = 462

    return config


# ============================================================
# QUESTION HANDLING
# ============================================================

def build_questions(samples, is_mc: bool):
    """
    Produce two parallel lists:

    final_queries:
        What Gemini sees when producing the final answer.

    retrieval_queries:
        What ComoRAG uses to search its memories.

    For EN.QA these are normally identical.

    For EN.MC:
        final_query       = question + A/B/C/D choices
        retrieval_query   = question only
    """

    final_queries = []
    retrieval_queries = []

    for i, sample in enumerate(samples):

        if "question" not in sample:
            raise KeyError(
                f"Sample {i} has no 'question' field."
            )

        final_query = sample["question"]

        if is_mc:
            if "retrieval_question" not in sample:
                raise KeyError(
                    "EN.MC sample is missing "
                    "'retrieval_question'. "
                    "The preprocessing script must store "
                    "the question without options there."
                )

            retrieval_query = sample["retrieval_question"]

        else:
            retrieval_query = sample.get(
                "retrieval_question",
                final_query
            )

        final_queries.append(final_query)
        retrieval_queries.append(retrieval_query)

    return final_queries, retrieval_queries


# ============================================================
# ONE CONTEXT / BOOK
# ============================================================

def process_context(
    context_dir: Path,
    task_name: str,
    is_mc: bool,
    base_config: BaseConfig,
    force: bool = False,
    rebuild_index: bool = False,
):
    """
    Index and answer all questions associated with one
    InfiniteBench context/book.
    """

    context_id = context_dir.name

    corpus_path = context_dir / "corpus.jsonl"
    qas_path = context_dir / "qas.jsonl"

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Missing corpus file: {corpus_path}"
        )

    if not qas_path.exists():
        raise FileNotFoundError(
            f"Missing question file: {qas_path}"
        )

    # --------------------------------------------------------
    # Result location is task-specific
    # --------------------------------------------------------

    result_dir = (
        RESULT_ROOT
        / task_name
        / context_id
    )

    results_file = result_dir / "results.json"

    if results_file.exists() and not force:
        print(
            f"[SKIP] {task_name}/{context_id}: "
            f"results already exist."
        )
        return

    # --------------------------------------------------------
    # Load corpus
    # --------------------------------------------------------

    corpus = load_jsonl(corpus_path)

    docs = [
        document["contents"]
        for document in corpus
    ]

    # --------------------------------------------------------
    # Load questions
    # --------------------------------------------------------

    samples = load_jsonl(qas_path)

    if not samples:
        print(
            f"[SKIP] {task_name}/{context_id}: "
            f"no questions."
        )
        return

    final_queries, retrieval_queries = build_questions(
        samples,
        is_mc=is_mc
    )

    # --------------------------------------------------------
    # Copy config for THIS book/context
    # --------------------------------------------------------

    config = copy.deepcopy(base_config)

    config.is_mc = is_mc

    # Avoid the bug from main_openai.py:
    #
    # WRONG:
    # config.corpus_len = len(corpus),
    #
    # RIGHT:
    config.corpus_len = len(corpus)

    # --------------------------------------------------------
    # SHARED INDEX LOCATION
    # --------------------------------------------------------
    #
    # Notice task_name is deliberately NOT included here.
    #
    # EN.QA:
    # outputs/infinitebench/shared_indexes/ABC123/
    #
    # EN.MC:
    # outputs/infinitebench/shared_indexes/ABC123/
    #
    # Same context hash => same ComoRAG index.
    # --------------------------------------------------------

    shared_index_dir = (
        SHARED_INDEX_ROOT
        / context_id
    )

    config.save_dir = str(shared_index_dir)

    # --------------------------------------------------------
    # TASK-SPECIFIC RESULT LOCATION
    # --------------------------------------------------------

    config.output_dir = str(result_dir)

    # We set this only as descriptive metadata.
    # save_dir/output_dir above determine the actual paths.
    config.dataset = None

    print()
    print("=" * 70)
    print(f"Task:              {task_name}")
    print(f"is_mc:             {config.is_mc}")
    print(f"Context:           {context_id}")
    print(f"Documents/chunks:  {len(corpus)}")
    print(f"Questions:         {len(samples)}")
    print(f"Shared index:      {config.save_dir}")
    print(f"Results:           {config.output_dir}")
    print("=" * 70)

    # --------------------------------------------------------
    # Shared-index marker
    # --------------------------------------------------------

    marker_path = get_index_marker_path(config)

    index_metadata = expected_index_metadata(
        context_id=context_id,
        corpus_path=corpus_path,
        corpus_len=len(corpus),
        config=config,
    )

    if rebuild_index:
        index_is_complete = False
        print(
            "[INDEX] --rebuild-index supplied; "
            "existing completion marker will be ignored."
        )
    else:
        index_is_complete = index_marker_is_valid(
            marker_path=marker_path,
            expected_metadata=index_metadata,
        )

    # --------------------------------------------------------
    # Instantiate ComoRAG
    # --------------------------------------------------------

    comorag = ComoRAG(
        global_config=config
    )

    # --------------------------------------------------------
    # Index
    # --------------------------------------------------------
    #
    # This can reuse existing files from the shared index.
    #
    # If EN.QA and EN.MC use the same context hash, the second
    # task sees the already-created embeddings/index.
    # --------------------------------------------------------

    if index_is_complete:

        print()
        print(
            f"[INDEX] Complete shared index found."
        )
        print(
            f"[INDEX] Reusing index for {context_id}"
        )
        print(
            f"[INDEX] Marker: {marker_path}"
        )

    else:

        print()
        print(
            f"[INDEX] No valid completion marker found."
        )
        print(
            f"[INDEX] Building/resuming index for {context_id}"
        )

        # IMPORTANT:
        #
        # If this call crashes, execution stops here and no
        # .index_complete.json marker gets written.
        #
        # Existing partial files may remain on disk. On the next
        # invocation ComoRAG can attempt to resume/reuse them,
        # but the runner will still treat the index as incomplete.
        #
        comorag.index(docs)

        # ---------------------------------------------
        # Check important artifacts before marking it
        # complete.
        # ---------------------------------------------

        artifacts_ok, missing_files = verify_index_artifacts(
            config
        )

        if not artifacts_ok:

            print()
            print(
                "[INDEX] ComoRAG.index() returned, "
                "but required artifacts are missing:"
            )

            for path in missing_files:
                print(
                    f"        MISSING: {path}"
                )

            raise RuntimeError(
                f"Index for {context_id} did not pass "
                "the completion check. "
                "No completion marker was written."
            )

    # Only reach this line after successful indexing AND
    # artifact validation.
        write_index_marker(
            marker_path=marker_path,
            metadata=index_metadata,
        )
    
    # --------------------------------------------------------
    # Answer questions
    # --------------------------------------------------------
    print()
    print(
        f"[QA] Answering {len(final_queries)} "
        f"{task_name} questions"
    )

    solutions = comorag.try_answer(
        final_queries,
        retrieval_queries=retrieval_queries
    )

    if len(solutions) != len(samples):
        raise RuntimeError(
            f"Expected {len(samples)} solutions but "
            f"ComoRAG returned {len(solutions)}."
        )

    # --------------------------------------------------------
    # Gold answers
    # --------------------------------------------------------

    gold_answers = get_gold_answers(samples)

    for idx, solution in enumerate(solutions):
        solution.gold_answers = list(
            gold_answers[idx]
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    result_list = []

    for idx, (
        final_query,
        retrieval_query,
        solution
    ) in enumerate(
        zip(
            final_queries,
            retrieval_queries,
            solutions
        )
    ):

        result_list.append(
            {
                "idx": idx,
                "context_id": context_id,
                "task": task_name,
                "is_mc": is_mc,

                # Final question seen by QA model
                "question": final_query,

                # Search query used by ComoRAG
                "retrieval_question": retrieval_query,

                "golden_answers": solution.gold_answers,

                # Raw ComoRAG/Gemini response
                "output": solution.answer,
            }
        )

    save_json(
        results_file,
        result_list
    )

    print(
        f"[DONE] Saved {len(result_list)} answers to:"
    )
    print(results_file)


# ============================================================
# AGGREGATE RESULTS
# ============================================================

def aggregate_task_results(task_name: str):
    """
    Combine the context-level results into one task-level file.
    """

    task_result_root = (
        RESULT_ROOT
        / task_name
    )

    all_results = []

    if not task_result_root.exists():
        return

    for results_file in sorted(
        task_result_root.glob("*/results.json")
    ):

        with results_file.open(
            "r",
            encoding="utf-8"
        ) as f:
            context_results = json.load(f)

        all_results.extend(context_results)

    aggregate_file = (
        task_result_root
        / "results_all.json"
    )

    save_json(
        aggregate_file,
        all_results
    )

    print()
    print(
        f"[AGGREGATE] {task_name}: "
        f"{len(all_results)} total answers"
    )
    print(
        f"[AGGREGATE] Saved to {aggregate_file}"
    )


# ============================================================
# RUN A TASK
# ============================================================

def run_task(
    task_name: str,
    is_mc: bool,
    base_config: BaseConfig,
    limit_contexts=None,
    force=False,
    rebuild_index=False
):
    """
    Run either EN.QA or EN.MC.
    """

    task_dir = (
        DATA_ROOT
        / task_name
    )

    if not task_dir.exists():
        raise FileNotFoundError(
            f"Task directory does not exist: {task_dir}"
        )

    context_dirs = sorted(
        directory
        for directory in task_dir.iterdir()
        if directory.is_dir()
    )

    if limit_contexts is not None:
        context_dirs = context_dirs[:limit_contexts]

    print()
    print("#" * 70)
    print(f"Starting task: {task_name}")
    print(f"is_mc = {is_mc}")
    print(f"Contexts found: {len(context_dirs)}")
    print("#" * 70)

    for number, context_dir in enumerate(
        context_dirs,
        start=1
    ):

        print()
        print(
            f"[{number}/{len(context_dirs)}] "
            f"{task_name}/{context_dir.name}"
        )

        process_context(
            context_dir=context_dir,
            task_name=task_name,
            is_mc=is_mc,
            base_config=base_config,
            force=force,
            rebuild_index=rebuild_index
        )

    aggregate_task_results(task_name)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run Gemini 3.1 Flash-Lite + ComoRAG "
            "on InfiniteBench EN.QA and EN.MC."
        )
    )

    parser.add_argument(
        "--task",
        choices=["enqa", "enmc", "all"],
        default="all",
        help=(
            "Task to run. "
            "'all' runs EN.QA followed by EN.MC."
        ),
    )

    parser.add_argument(
        "--limit-contexts",
        type=int,
        default=None,
        help=(
            "Only run the first N context directories. "
            "Useful for pilot tests."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-run contexts even if results.json "
            "already exists."
        ),
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help=(
            "Ignore existing index-completion markers "
            "and rebuild/resume the shared indexes."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load API key
    # --------------------------------------------------------

    env_path = PROJECT_ROOT / ".env"

    load_dotenv(
        dotenv_path=env_path
    )

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY was not found. "
            f"Expected it in {env_path}"
        )

    print(
        "Gemini API key found: True"
    )

    # --------------------------------------------------------
    # Create root directories
    # --------------------------------------------------------

    SHARED_INDEX_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Shared base configuration
    # --------------------------------------------------------

    base_config = make_base_config(
        api_key
    )

    # --------------------------------------------------------
    # EN.QA
    # --------------------------------------------------------

    if args.task in (
        "enqa",
        "all",
    ):

        run_task(
            task_name="enqa",
            is_mc=False,
            base_config=base_config,
            limit_contexts=args.limit_contexts,
            force=args.force,
            rebuild_index=args.rebuild_index
        )

    # --------------------------------------------------------
    # EN.MC
    # --------------------------------------------------------

    if args.task in (
        "enmc",
        "all",
    ):

        run_task(
            task_name="enmc",
            is_mc=True,
            base_config=base_config,
            limit_contexts=args.limit_contexts,
            force=args.force,
            rebuild_index=args.rebuild_index
        )


if __name__ == "__main__":
    main()