import json
import time
from pathlib import Path

import polars as pl
from openai import OpenAI

TEACHER_SYSTEM_PROMPT = (
    "You are a helpful assistant that generates question-answer pairs "
    "based on the provided document content. Generate questions that are "
    "answerable only from the given content. Each question should be clear "
    "and specific. Each answer should be comprehensive and accurate based "
    "on the content provided."
)

TEACHER_USER_PROMPT_TEMPLATE = (
    "Based on the following content, generate {num_examples} question-answer pairs. "
    'Output them as a JSON object with a single key "qa_pairs" containing an array '
    'of objects with "question" and "answer" fields.\n\n'
    "Content:\n{content}"
)

DEFAULT_MIN_QUESTION_WORDS = 5
DEFAULT_MIN_ANSWER_WORDS = 20
MIN_CHUNK_WORDS = 50
API_CALL_DELAY_SECONDS = 0.5
OUTPUT_PARQUET_NAME = "generated_qa.parquet"


def generate_qa_pairs(
    content_chunk: str,
    client: OpenAI,
    model: str,
    num_examples: int,
) -> list[dict]:
    prompt = TEACHER_USER_PROMPT_TEMPLATE.format(
        num_examples=num_examples, content=content_chunk
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content or "{}"
    parsed = json.loads(raw)
    qa_pairs = parsed.get("qa_pairs", [])

    return [
        {"question": pair["question"], "answer": pair["answer"]}
        for pair in qa_pairs
        if "question" in pair and "answer" in pair
    ]


def filter_qa_pairs(
    examples: list[dict],
    min_question_words: int = DEFAULT_MIN_QUESTION_WORDS,
    min_answer_words: int = DEFAULT_MIN_ANSWER_WORDS,
) -> list[dict]:
    return [
        ex
        for ex in examples
        if len(ex.get("question", "").split()) >= min_question_words
        and len(ex.get("answer", "").split()) >= min_answer_words
    ]


def _build_chunks(df: pl.DataFrame, chunk_size: int) -> list[dict]:
    rows = df.sort("source_file", "page_number").iter_rows(named=True)
    chunks: list[dict] = []
    current_lines: list[str] = []
    current_meta: dict | None = None
    current_word_count = 0

    def flush():
        if current_lines and current_meta:
            text = "\n".join(current_lines)
            chunks.append(
                {
                    "text": f"[Source: {current_meta['source_file']}, Page: {current_meta['page_number']}]\n{text}",
                    "word_count": current_word_count,
                }
            )

    for row in rows:
        line = row["text_content"].strip()
        if not line:
            continue

        line_words = len(line.split())
        source = row["source_file"]
        page = row["page_number"]
        section = row["section_type"]

        should_split = (section in ("title", "section_header") and current_lines) or (
            current_meta is not None and current_word_count + line_words > chunk_size
        )
        if should_split:
            flush()
            current_lines = []
            current_word_count = 0

        if not current_lines:
            current_meta = {"source_file": source, "page_number": page}
        current_lines.append(line)
        current_word_count += line_words

    flush()
    filtered = [c for c in chunks if c["word_count"] >= MIN_CHUNK_WORDS]
    if len(filtered) < len(chunks):
        print(
            f"Filtered {len(chunks) - len(filtered)}/{len(chunks)} chunks "
            f"below {MIN_CHUNK_WORDS} words"
        )
    return filtered


def generate_dataset(
    input_parquet: str,
    output_dir: str,
    client: OpenAI,
    model: str,
    num_examples: int,
    chunk_size: int = 2000,
) -> str:
    df = pl.read_parquet(input_parquet)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    chunks = _build_chunks(df, chunk_size)

    if not chunks:
        raise RuntimeError("No text chunks produced from input parquet")

    total_chunk_words = sum(c["word_count"] for c in chunks)
    print(
        f"Generating {num_examples} QA pairs from {len(chunks)} chunks ({total_chunk_words:,} total words)"
    )

    all_qa: list[dict] = []
    fail_count = 0

    for i, chunk in enumerate(chunks):
        proportional = max(
            1, round(num_examples * chunk["word_count"] / total_chunk_words)
        )

        try:
            pairs = generate_qa_pairs(chunk["text"], client, model, proportional)
            all_qa.extend(pairs)
            print(
                f"Chunk {i + 1}/{len(chunks)}: generated {len(pairs)} pairs (requested {proportional})"
            )
        except Exception as e:
            fail_count += 1
            print(f"Chunk {i + 1}/{len(chunks)} failed: {e}")

        time.sleep(API_CALL_DELAY_SECONDS)

    filtered = filter_qa_pairs(all_qa)
    removed = len(all_qa) - len(filtered)

    print(
        f"Generated {len(all_qa)} total, {len(filtered)} after filtering "
        f"({removed} removed by min word count, {fail_count} chunk failures)"
    )

    if len(filtered) == 0:
        raise RuntimeError(
            "Zero QA pairs generated. Check API connectivity and teacher model."
        )

    if len(filtered) < num_examples * 0.5:
        raise RuntimeError(
            f"Only {len(filtered)}/{num_examples} QA pairs generated "
            f"({len(filtered) / num_examples * 100:.0f}%). "
            f"Training data insufficient — check source documents or reduce --num-examples."
        )

    result_df = pl.DataFrame(
        {
            "instruction": [qa["question"] for qa in filtered],
            "output": [qa["answer"] for qa in filtered],
        },
        schema={"instruction": pl.String, "output": pl.String},
    )

    parquet_path = output_path / OUTPUT_PARQUET_NAME
    result_df.write_parquet(parquet_path)
    print(f"Wrote {len(result_df)} QA pairs to {parquet_path}")

    return str(parquet_path)
