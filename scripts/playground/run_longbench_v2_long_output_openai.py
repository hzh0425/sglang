#!/usr/bin/env python3
"""Run multiple ultra-long LongBench-v2 samples through an OpenAI-compatible chat API.

This script picks extra-long samples from the local LongBench-v2 subset and
rewrites each multiple-choice task into a long-form analysis prompt so the model
is encouraged to produce long answers.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


DEFAULT_DATA_PATH = Path("data/longbench_v2/ultra_long_p90.json")
PREFERRED_DOMAINS = [
    "Code Repository Understanding",
    "Multi-Document QA",
    "Long In-context Learning",
    "Single-Document QA",
    "Long Structured Data Understanding",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select multiple ultra-long LongBench-v2 samples and run long-form "
            "generation through an OpenAI-compatible chat endpoint."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to a local LongBench-v2 JSON/JSONL file.",
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default="",
        help="Specific sample _id to use. Empty means auto-select.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="",
        help="Only consider samples from this domain.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Start index within the filtered candidate list.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of samples to run. Ignored when --sample-id is set.",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default="EMPTY",
        help="OpenAI-compatible API key.",
    )
    parser.add_argument(
        "--openai-api-base",
        type=str,
        default="http://localhost:8188/v1",
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model name. Empty means auto-pick the first served model.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="",
        help="Tokenizer path/name. Empty means reuse the selected model name.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=4096,
        help="Maximum completion tokens for the chat request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--min-output-words",
        type=int,
        default=1200,
        help="Ask the model to produce at least this many words if possible.",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=100000,
        help="Truncate context to at most this many tokens. Set <=0 to disable.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=0,
        help="Further truncate context to at most this many characters. Set <=0 to disable.",
    )
    parser.add_argument(
        "--save-response",
        type=Path,
        default=None,
        help="Optional path to save the final response text.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Failed to parse JSONL line {line_no} in {path}: {exc}"
                    ) from exc
        return rows

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"Expected top-level list in {path}, got {type(data)!r}")
    return data


def get_choice(row: dict[str, Any], letter: str) -> str:
    return str(row.get(f"choice_{letter}", row.get(letter, "")))


def build_candidate_rows(
    rows: list[dict[str, Any]],
    domain: str,
    index: int,
) -> list[dict[str, Any]]:
    candidates = rows
    if domain:
        candidates = [row for row in candidates if row.get("domain") == domain]
        if not candidates:
            raise ValueError(f"No samples found for domain: {domain}")

    candidates = sorted(
        candidates, key=lambda row: len(row.get("context", "")), reverse=True
    )

    if not candidates:
        raise ValueError("No candidate rows available after filtering.")
    if index < 0 or index >= len(candidates):
        raise IndexError(f"Index {index} out of range for {len(candidates)} candidates.")
    return candidates[index:]


def select_rows(
    rows: list[dict[str, Any]],
    sample_id: str,
    domain: str,
    index: int,
    num_samples: int,
) -> list[dict[str, Any]]:
    if sample_id:
        for row in rows:
            if str(row.get("_id", "")) == sample_id:
                return [row]
        raise ValueError(f"Sample id not found: {sample_id}")

    candidates = build_candidate_rows(rows, domain, index)
    if num_samples <= 0:
        raise ValueError("--num-samples must be > 0")

    if domain:
        return candidates[:num_samples]

    # Prefer diversity across domains first, then fill remaining slots by length.
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    for preferred_domain in PREFERRED_DOMAINS:
        for row in candidates:
            row_id = str(row.get("_id", ""))
            if row.get("domain") == preferred_domain and row_id not in selected_ids:
                selected.append(row)
                selected_ids.add(row_id)
                break
        if len(selected) >= num_samples:
            return selected[:num_samples]

    for row in candidates:
        row_id = str(row.get("_id", ""))
        if row_id not in selected_ids:
            selected.append(row)
            selected_ids.add(row_id)
        if len(selected) >= num_samples:
            break

    return selected[:num_samples]


def truncate_context_by_tokens(
    context: str, tokenizer: Any, max_context_tokens: int
) -> tuple[str, bool, int, int]:
    token_ids = tokenizer.encode(context, add_special_tokens=False)
    original_tokens = len(token_ids)
    if max_context_tokens <= 0 or original_tokens <= max_context_tokens:
        return context, False, original_tokens, original_tokens

    head_tokens = max_context_tokens // 2
    tail_tokens = max_context_tokens - head_tokens
    head_text = tokenizer.decode(token_ids[:head_tokens], skip_special_tokens=True)
    tail_text = tokenizer.decode(token_ids[-tail_tokens:], skip_special_tokens=True)
    truncated = (
        head_text
        + "\n\n[... middle content truncated to fit the model context window ...]\n\n"
        + tail_text
    )
    used_tokens = len(tokenizer.encode(truncated, add_special_tokens=False))
    return truncated, True, original_tokens, used_tokens


def truncate_context_by_chars(context: str, max_context_chars: int) -> tuple[str, bool]:
    if max_context_chars <= 0 or len(context) <= max_context_chars:
        return context, False

    head_chars = max_context_chars // 2
    tail_chars = max_context_chars - head_chars
    truncated = (
        context[:head_chars]
        + "\n\n[... middle content truncated to fit the character limit ...]\n\n"
        + context[-tail_chars:]
    )
    return truncated, True


def build_long_output_prompt(
    row: dict[str, Any],
    tokenizer: Any,
    min_output_words: int,
    max_context_tokens: int,
    max_context_chars: int,
) -> tuple[str, bool, bool, int, int, int, int]:
    question = str(row.get("question", "")).strip()
    choice_a = get_choice(row, "A").strip()
    choice_b = get_choice(row, "B").strip()
    choice_c = get_choice(row, "C").strip()
    choice_d = get_choice(row, "D").strip()
    original_context = str(row.get("context", "")).strip()
    original_context_chars = len(original_context)

    context, token_truncated, original_context_tokens, used_context_tokens = (
        truncate_context_by_tokens(original_context, tokenizer, max_context_tokens)
    )
    context, char_truncated = truncate_context_by_chars(context, max_context_chars)
    used_context_chars = len(context)
    if char_truncated:
        used_context_tokens = len(tokenizer.encode(context, add_special_tokens=False))

    domain = str(row.get("domain", "unknown"))
    difficulty = str(row.get("difficulty", "unknown"))

    prompt = f"""You are given an ultra-long LongBench-v2 task.

Please read the full context carefully and answer in a long-form way.

Task metadata:
- Domain: {domain}
- Difficulty: {difficulty}
- Context token-truncated: {"yes" if token_truncated else "no"}
- Context char-truncated: {"yes" if char_truncated else "no"}

Question:
{question}

Options:
(A) {choice_a}
(B) {choice_b}
(C) {choice_c}
(D) {choice_d}

Required output format:
1. A concise restatement of the task.
2. A detailed step-by-step analysis of the relevant evidence from the context.
3. A separate discussion of each option A/B/C/D, including why it is correct or incorrect.
4. A final conclusion section.
5. End with exactly one line in the format: Final answer: (X)

Important constraints:
- Produce a detailed long answer instead of a short answer.
- Aim for at least {min_output_words} words if the model and max token limit allow.
- Ground the reasoning in the supplied context only.
- Do not skip the option-by-option analysis.

Context:
<context>
{context}
</context>
"""
    return (
        prompt,
        token_truncated,
        char_truncated,
        original_context_tokens,
        used_context_tokens,
        original_context_chars,
        used_context_chars,
    )


def get_model_name(client: OpenAI, model_name: str) -> str:
    if model_name:
        return model_name
    models = client.models.list()
    if not models.data:
        raise RuntimeError("No models available from the OpenAI-compatible endpoint.")
    return models.data[0].id


def load_tokenizer(tokenizer_path: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Please install transformers to enable token-based truncation: "
            "pip install transformers"
        ) from exc

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def run_text_only(
    client: OpenAI,
    model: str,
    prompt: str,
    max_completion_tokens: int,
    temperature: float,
) -> tuple[str, float, Any]:
    start_time = time.time()
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
    )
    cost_time = time.time() - start_time
    text = chat_completion.choices[0].message.content if chat_completion.choices else ""
    return text or "", cost_time, chat_completion.usage


def main() -> int:
    args = parse_args()
    rows = load_rows(args.data_path)
    selected_rows = select_rows(
        rows=rows,
        sample_id=args.sample_id,
        domain=args.domain,
        index=args.index,
        num_samples=args.num_samples,
    )

    client = OpenAI(
        api_key=args.openai_api_key,
        base_url=args.openai_api_base,
    )
    model = get_model_name(client, args.model)
    tokenizer_path = args.tokenizer_path or model
    tokenizer = load_tokenizer(tokenizer_path)
    print(f"[Info] Selected {len(selected_rows)} sample(s)")
    print(f"  model: {model}")
    print(f"  tokenizer_path: {tokenizer_path}")
    print(f"  max_completion_tokens: {args.max_completion_tokens}")

    saved_records: list[dict[str, Any]] = []

    for sample_idx, row in enumerate(selected_rows, 1):
        (
            prompt,
            token_truncated,
            char_truncated,
            original_context_tokens,
            used_context_tokens,
            original_context_chars,
            used_context_chars,
        ) = build_long_output_prompt(
            row=row,
            tokenizer=tokenizer,
            min_output_words=args.min_output_words,
            max_context_tokens=args.max_context_tokens,
            max_context_chars=args.max_context_chars,
        )
        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))

        print(f"\n===== Sample {sample_idx}/{len(selected_rows)} =====")
        print(f"  id: {row.get('_id')}")
        print(f"  domain: {row.get('domain')}")
        print(f"  difficulty: {row.get('difficulty')}")
        print(f"  original_context_chars: {original_context_chars}")
        print(f"  used_context_chars: {used_context_chars}")
        print(f"  original_context_tokens: {original_context_tokens}")
        print(f"  used_context_tokens: {used_context_tokens}")
        print(f"  context_token_truncated: {token_truncated}")
        print(f"  context_char_truncated: {char_truncated}")
        print(f"  prompt_tokens: {prompt_tokens}")

        response_text, latency_s, usage = run_text_only(
            client=client,
            model=model,
            prompt=prompt,
            max_completion_tokens=args.max_completion_tokens,
            temperature=args.temperature,
        )

        print(f"[Done] latency_s={latency_s:.3f}")
        prompt_tokens_usage = None
        completion_tokens = None
        total_tokens = None
        if usage is not None:
            prompt_tokens_usage = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
            print(
                "[Usage] "
                f"prompt_tokens={prompt_tokens_usage} "
                f"completion_tokens={completion_tokens} "
                f"total_tokens={total_tokens}"
            )

        saved_records.append(
            {
                "sample_index": sample_idx,
                "id": row.get("_id"),
                "domain": row.get("domain"),
                "difficulty": row.get("difficulty"),
                "original_context_chars": original_context_chars,
                "used_context_chars": used_context_chars,
                "original_context_tokens": original_context_tokens,
                "used_context_tokens": used_context_tokens,
                "context_token_truncated": token_truncated,
                "context_char_truncated": char_truncated,
                "prompt_tokens_estimate": prompt_tokens,
                "latency_s": latency_s,
                "usage_prompt_tokens": prompt_tokens_usage,
                "usage_completion_tokens": completion_tokens,
                "usage_total_tokens": total_tokens,
                "response": response_text,
            }
        )

        print("\n===== Response Begin =====\n")
        print(response_text)
        print("\n===== Response End =====")

    if args.save_response is not None:
        args.save_response.parent.mkdir(parents=True, exist_ok=True)
        if len(saved_records) == 1:
            args.save_response.write_text(saved_records[0]["response"], encoding="utf-8")
        else:
            with args.save_response.open("w", encoding="utf-8") as f:
                for record in saved_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\n[Saved] responses -> {args.save_response}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
