from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
from jinja2 import Template
from qwen_vl_utils.vision_process import fetch_video
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = Path(__file__).resolve().parent / "templates"
FORMAT_PROMPT = TEMPLATE_ROOT / "quantiphy_video.jinja"
CHAT_TEMPLATE = TEMPLATE_ROOT / "qwen3_5_no_think.jinja"

SYSTEM_PROMPT = (
    "You are an expert video analyst specializing in physics measurements. "
    "Analyze the video frames carefully and provide ONLY the numerical answer with units. "
    "No explanation or reasoning needed. Format your response as: [value] [unit]. "
    "Example: 2.5 cm. Be as accurate as possible with measurements and calculations. "
    "Please give me an estimated answer even if you are not sure."
)

MODEL_SETTINGS = {
    "4b": {"gpu_memory_utilization": 0.2, "enable_chunked_prefill": False},
    "9b": {"gpu_memory_utilization": 0.5, "enable_chunked_prefill": True},
}

MAX_PROMPT_LENGTH = 4096
MAX_RESPONSE_LENGTH = 512
MAX_MODEL_LEN = MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH
VIDEO_NFRAMES = 16
VIDEO_TIMESTAMP_FPS = 24.0
VIDEO_FPS = 2.0
MIN_PIXELS = 0
MAX_PIXELS = 262144
SEED = 1
DEFAULT_VIDEO_EXT = ".mp4"

SAMPLING_CONFIG = {
    "temperature": 0.01,
    "top_p": 0.001,
    "top_k": -1,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "max_tokens": MAX_RESPONSE_LENGTH,
    "n": 1,
}

MRA_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
NFRAMES_INTERVAL_PATTERN = re.compile(
    r"nframes should in interval \[(\d+), (\d+)\], but got (\d+)"
)
PREDICTION_COLUMNS = {"raw_response", "parsed_value", "model"}
PREDICTION_COLUMN_KEYS = {column.casefold() for column in PREDICTION_COLUMNS}


_GIVEN_THAT_RE_TEMPLATE = r"^\s*Given\s+that\s+{}\s*[,.;:]\s*"
DEPTH_PREFIX = (
    "Additionally, you have the following information about the distance between the objects "
    "in the video and the shooting camera:"
)


def _clean_text(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def _normalise_header(value: Any) -> str:
    return str(value or "").lstrip("\ufeff").strip()


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _category_from_fields(inference_type: Any, video_type: Any) -> str:
    inference = _clean_text(inference_type).upper()
    video = _clean_text(video_type).upper()
    if len(inference) < 1 or len(video) < 2:
        return ""
    prefix, dimension = inference[0], video[1]
    return f"{prefix}{dimension}" if prefix in {"S", "D"} and dimension in {"2", "3"} else ""


def _resolve_video_path(
    video_value: Any,
    video_id: str,
    video_dir: Path | None,
    video_ext: str,
) -> str:
    value = _clean_text(video_value)
    if value.startswith(("http://", "https://", "s3://", "file://")):
        return value
    if value:
        path = Path(value).expanduser()
        if not path.is_absolute() and video_dir is not None:
            path = video_dir / path
        return str(path.resolve())
    if video_dir is None:
        raise ValueError(
            f"CSV row for video_id={video_id!r} has no video path; pass --video-dir"
        )
    suffix = video_ext if video_ext.startswith(".") else f".{video_ext}"
    return str((video_dir / f"{video_id}{suffix}").resolve())


def _load_csv(
    path: Path,
    video_dir: Path | None,
    video_ext: str,
) -> tuple[list[dict[str, Any]], list[list[str]], list[tuple[str, int]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"input CSV is empty: {path}") from exc
        if not any(_normalise_header(value) for value in header):
            raise ValueError(f"input CSV has no column names: {path}")

        positions: dict[str, int] = {}
        for index, value in enumerate(header):
            name = _normalise_header(value).casefold()
            if name and name not in positions:
                positions[name] = index

        missing = [name for name in ("video_id", "question") if name not in positions]
        if missing:
            raise ValueError(
                f"input CSV is missing required column(s): {', '.join(missing)}"
            )

        output_columns: list[tuple[str, int]] = []
        seen: set[str] = set()
        for index, value in enumerate(header):
            name = _normalise_header(value)
            if index == 0 and not name:
                name = "id"
            if not name or name.casefold() in PREDICTION_COLUMN_KEYS:
                continue
            key = name.casefold()
            if key in seen:
                continue
            seen.add(key)
            output_columns.append((name, index))

        def cell(row: list[str], *names: str) -> str:
            for name in names:
                index = positions.get(name.casefold())
                if index is not None and index < len(row):
                    value = _clean_text(row[index])
                    if value:
                        return value
            return ""

        records: list[dict[str, Any]] = []
        source_rows: list[list[str]] = []
        for row_number, row in enumerate(reader, start=2):
            if not any(_clean_text(value) for value in row):
                continue
            video_id = cell(row, "video_id")
            question = cell(row, "question", "raw_question")
            if not video_id or not question:
                raise ValueError(
                    f"CSV row {row_number} must contain video_id and question"
                )
            video_source = cell(row, "video_source")
            video_type = cell(row, "video_type")
            inference_type = cell(row, "inference_type")
            prior = cell(row, "ground_truth_prior", "prior")
            depth_info = cell(row, "depth_info")
            answer = cell(
                row,
                "ground_truth_posterior",
                "answer",
                "ground_truth",
                "gt",
            )
            original_id = _clean_text(row[0] if row else "") or str(len(records))
            record = {
                "id": f"quantiphy_{len(records)}",
                "original_id": original_id,
                "video": _resolve_video_path(
                    cell(row, "video", "video_path", "video_file"),
                    video_id,
                    video_dir,
                    video_ext,
                ),
                "question": question,
                "raw_question": question,
                "answer": answer,
                "video_id": video_id,
                "video_source": video_source,
                "video_type": video_type,
                "fps": cell(row, "fps"),
                "inference_type": inference_type,
                "ground_truth_prior": prior,
                "depth_info": depth_info,
                "category": _category_from_fields(inference_type, video_type),
            }
            records.append(record)
            source_rows.append(row)

    if not records:
        raise ValueError(f"input CSV contains no data rows: {path}")
    return records, source_rows, output_columns


def _load_model_tools(model_path: Path):
    chat_template = CHAT_TEMPLATE.read_text(encoding="utf-8")
    kwargs = {"trust_remote_code": True, "use_fast": True}
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), **kwargs)
    processor = AutoProcessor.from_pretrained(str(model_path), **kwargs)
    tokenizer.chat_template = chat_template
    processor.chat_template = chat_template
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, processor


def _normalise_question(value: Any) -> str:
    question = _clean_text(value).replace("？", "?")
    return question if not question or question[-1] in ".!?" else question + "?"


def _content_from_record(record: dict[str, Any]) -> str:
    existing = _clean_text(record.get("content"))
    if existing:
        return existing
    prior = _clean_text(record.get("ground_truth_prior"))
    question = _clean_text(record.get("raw_question") or record.get("question"))
    if prior and question:
        pattern = _GIVEN_THAT_RE_TEMPLATE.format(re.escape(prior))
        stripped = re.sub(pattern, "", question, count=1, flags=re.IGNORECASE).strip()
        if stripped != question and stripped[:1].islower():
            stripped = stripped[:1].upper() + stripped[1:]
        question = stripped
    question = _normalise_question(question)
    parts: list[str] = []
    if prior:
        parts.append(f"Given that {prior}.")
    depth = _clean_text(record.get("depth_info"))
    if depth:
        parts.append(f"{DEPTH_PREFIX} {depth}")
    prefix = " ".join(parts).rstrip()
    if prefix and prefix[-1] not in ".!?":
        prefix += "."
    return f"{prefix} {question}".strip() if prefix else question


def _format_prompt(record: dict[str, Any], format_prompt: Template) -> str:
    return format_prompt.render(content=_content_from_record(record))


def _video_values(record: dict[str, Any]) -> list[str]:
    value = record.get("video")
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [] if value is None else [str(value)]


def _message_content(record: dict[str, Any], format_prompt: Template) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for index, text in enumerate(_format_prompt(record, format_prompt).split("<video>")):
        if index:
            content.append({"type": "video"})
        if text:
            content.append({"type": "text", "text": text})
    return content


def _record_metadata(record: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "id": record.get("id", index),
        "original_id": record.get("original_id", ""),
        "video_id": record.get("video_id", ""),
        "question": record.get("raw_question") or record.get("question", ""),
        "ground_truth": str(record.get("answer", "")),
        "inference_type": record.get("inference_type", ""),
        "video_type": record.get("video_type", ""),
        "video_source": record.get("video_source", ""),
        "category": record.get("category", ""),
    }


def _build_prompt_ids(
    example: dict[str, Any], tokenizer: Any, processor: Any, format_prompt: Template
) -> list[int]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _message_content(example, format_prompt)},
    ]
    rendered = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    return tokenizer.encode(rendered, add_special_tokens=False)[:MAX_PROMPT_LENGTH]


def _process_video(video: str) -> tuple[Any, float]:
    if not video.startswith(("http://", "https://", "s3://", "file://")) and not os.path.isfile(video):
        raise FileNotFoundError(f"video not found: {video}")
    vision_info = {
        "video": video,
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "video_fps": VIDEO_FPS,
        "nframes": VIDEO_NFRAMES,
    }
    try:
        return fetch_video(
            vision_info,
            return_video_sample_fps=True,
            return_video_metadata=True,
        )
    except ValueError as exc:
        match = NFRAMES_INTERVAL_PATTERN.search(str(exc))
        if match is None:
            raise
        min_allowed, max_allowed, requested = (int(value) for value in match.groups())
        if requested <= max_allowed or max_allowed < min_allowed:
            raise
        vision_info["nframes"] = max_allowed
        return fetch_video(
            vision_info,
            return_video_sample_fps=True,
            return_video_metadata=True,
        )


def _video_input(example: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    processed_videos: list[Any] = []
    metadata_list: list[dict[str, Any]] = []
    timestamp_fps = _positive_float(example.get("fps")) or VIDEO_TIMESTAMP_FPS
    valid_metadata_keys = {
        "total_num_frames",
        "fps",
        "width",
        "height",
        "duration",
        "video_backend",
        "frames_indices",
    }
    for video in _video_values(example):
        video_out, sample_fps = _process_video(video)
        metadata: dict[str, Any] = {}
        if isinstance(video_out, tuple) and len(video_out) == 2:
            video_input, raw_metadata = video_out
            if raw_metadata is not None:
                metadata = {
                    key: value
                    for key, value in dict(raw_metadata).items()
                    if key in valid_metadata_keys
                }
        else:
            video_input = video_out
        processor_fps = timestamp_fps or float(sample_fps)
        processor_fps = float(processor_fps if processor_fps > 0 else 24.0)
        num_frames = int(video_input.shape[0]) if hasattr(video_input, "shape") else len(video_input)
        frame_indices = metadata.get("frames_indices")
        if hasattr(frame_indices, "detach"):
            frame_indices = frame_indices.detach().cpu().tolist()
        elif frame_indices is not None:
            frame_indices = list(frame_indices)
        if not frame_indices or len(frame_indices) != num_frames:
            frame_indices = list(range(num_frames))
        metadata["fps"] = processor_fps
        metadata["frames_indices"] = frame_indices
        metadata["total_num_frames"] = int(metadata.get("total_num_frames", num_frames))
        processed_videos.append(video_input)
        metadata_list.append(metadata)
    return {"video": list(zip(processed_videos, metadata_list))}, {
        "fps": timestamp_fps,
        "do_sample_frames": False,
    }


def _build_inputs(
    records: list[dict[str, Any]],
    tokenizer: Any,
    processor: Any,
    format_prompt: Template,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inputs: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        multimodal_data, processor_kwargs = _video_input(record)
        inputs.append(
            {
                "prompt_token_ids": _build_prompt_ids(record, tokenizer, processor, format_prompt),
                "multi_modal_data": multimodal_data,
                "mm_processor_kwargs": processor_kwargs,
            }
        )
        metadata.append(_record_metadata(record, index))
        if (index + 1) % 20 == 0:
            print(f"[eval] preprocessed {index + 1}/{len(records)}", flush=True)
    return inputs, metadata


def _ensure_vocab_size(model_path: Path) -> None:
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    if hasattr(config, "vocab_size"):
        return
    text_config = getattr(config, "text_config", None)
    if text_config is None or not hasattr(text_config, "vocab_size"):
        return
    config_class = config.__class__
    if not hasattr(config_class, "vocab_size"):
        setattr(
            config_class,
            "vocab_size",
            property(lambda instance: getattr(instance.text_config, "vocab_size", None)),
        )


def _build_engine(model: str, model_path: Path) -> LLM:
    settings = MODEL_SETTINGS[model]
    _ensure_vocab_size(model_path)
    print(f"[eval] loading {model} from {model_path}", flush=True)
    return LLM(
        model=str(model_path),
        skip_tokenizer_init=False,
        trust_remote_code=True,
        dtype="bfloat16",
        seed=SEED,
        max_model_len=MAX_MODEL_LEN,
        tensor_parallel_size=1,
        gpu_memory_utilization=settings["gpu_memory_utilization"],
        max_num_batched_tokens=8192,
        max_num_seqs=128,
        enforce_eager=False,
        disable_custom_all_reduce=True,
        enable_chunked_prefill=settings["enable_chunked_prefill"],
        mm_processor_cache_gb=0,
    )


def _logit_bias(processor: Any) -> dict[int, float] | None:
    token_ids: list[int] = []
    for attribute in ("image_token", "video_token"):
        token = getattr(processor, attribute, None)
        if token is None:
            continue
        token_id = processor.tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            token_ids.append(token_id)
    return {token_id: -100 for token_id in token_ids} or None


def _strip_answer_tags(value: Any) -> str:
    text = str(value or "")
    match = re.search(
        r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL
    )
    return match.group(1).strip() if match else text.strip()


def _parse_prediction(value: Any) -> float | None:
    matches = NUMBER_PATTERN.findall(_strip_answer_tags(value))
    if not matches:
        return None
    try:
        return float(matches[0])
    except ValueError:
        return None


def _mra(response: str, ground_truth: str) -> float:
    prediction = _parse_prediction(response)
    gold = _parse_prediction(ground_truth)
    if prediction is None or gold is None or gold == 0:
        return 0.0
    relative_error = abs(prediction - gold) / max(abs(gold), 1e-9)
    return sum(relative_error < (1 - threshold) for threshold in MRA_THRESHOLDS) / len(MRA_THRESHOLDS)


def _category(record: dict[str, Any]) -> str | None:
    category = _category_from_fields(record.get("inference_type"), record.get("video_type"))
    if category:
        return category
    for key in ("category", "video_type", "inference_type"):
        match = re.search(r"[23][SD]|[SD][23]", str(record.get(key) or "").upper())
        if match:
            value = match.group(0)
            return value if value[0] in {"S", "D"} else f"{value[1]}{value[0]}"
    return None


def _normalise_finish_reason(value: Any) -> str:
    if value is None:
        return "unknown"
    return str(getattr(value, "value", value))


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return getattr(value, "value", str(value))


def _percentile(values: list[int], percentile: int) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _format_number(value: float | None) -> str:
    return "" if value is None else format(value, ".15g")


def _write_predictions_csv(
    path: Path,
    output_columns: list[tuple[str, int]],
    source_rows: list[list[str]],
    records: list[dict[str, Any]],
    model: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [name for name, _ in output_columns] + [
        "raw_response",
        "parsed_value",
        "model",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row, record in zip(source_rows, records):
            output = {
                name: row[index] if index < len(row) else ""
                for name, index in output_columns
            }
            output.update(
                {
                    "raw_response": record["response"],
                    "parsed_value": _format_number(record.get("parsed_value")),
                    "model": model,
                }
            )
            writer.writerow(output)


def evaluate(
    model: str,
    model_path: Path,
    input_csv: Path,
    video_dir: Path | None,
    video_ext: str,
    output_csv: Path,
    out_json: Path,
    generations_log: Path,
) -> dict[str, Any]:
    records, source_rows, output_columns = _load_csv(input_csv, video_dir, video_ext)
    format_prompt = Template(FORMAT_PROMPT.read_text(encoding="utf-8").strip())
    tokenizer, processor = _load_model_tools(model_path)
    print(f"[eval] preprocessing {len(records)} records from {input_csv}", flush=True)
    vllm_inputs, metadata = _build_inputs(records, tokenizer, processor, format_prompt)
    engine = _build_engine(model, model_path)
    logit_bias = _logit_bias(processor)

    sampling = SamplingParams(
        **SAMPLING_CONFIG,
        seed=SEED,
        detokenize=True,
        logit_bias=logit_bias,
    )
    print(f"[eval] generating {len(vllm_inputs)} records with seed={SEED}", flush=True)
    outputs = engine.generate(vllm_inputs, sampling)

    categories = ("S2", "D2", "S3", "D3")
    per_category = {category: [] for category in categories}
    all_scores: list[float] = []
    generation_records: list[dict[str, Any]] = []
    for output, sample_metadata in zip(outputs, metadata):
        completion = output.outputs[0]
        response = completion.text
        parsed_value = _parse_prediction(response)
        score = _mra(response, sample_metadata["ground_truth"])
        category = _category(sample_metadata)
        all_scores.append(score)
        if category in per_category:
            per_category[category].append(score)
        generation_records.append(
            {
                "seed": SEED,
                **sample_metadata,
                "response": response,
                "parsed_value": parsed_value,
                "finish_reason": _normalise_finish_reason(completion.finish_reason),
                "stop_reason": _json_safe(completion.stop_reason),
                "num_generated_tokens": len(completion.token_ids or []),
                "has_think_end": "</think>" in response.lower(),
                "mra": score,
            }
        )

    category_means = {
        category: (float(np.mean(values)) if values else None)
        for category, values in per_category.items()
    }
    overall = float(np.mean(all_scores)) if all_scores else None
    valid_category_means = [value for value in category_means.values() if value is not None]
    macro = (
        float(np.mean(valid_category_means))
        if len(valid_category_means) == len(categories)
        else None
    )
    finish_counts = Counter(record["finish_reason"] for record in generation_records)
    token_counts = [record["num_generated_tokens"] for record in generation_records]

    _write_predictions_csv(
        output_csv,
        output_columns,
        source_rows,
        generation_records,
        model,
    )
    settings = MODEL_SETTINGS[model]
    result: dict[str, Any] = {
        "model": model,
        "model_path": str(model_path),
        "input_csv": str(input_csv),
        "predictions_csv": str(output_csv),
        "input_format": "quantiphy_csv",
        "system_prompt": SYSTEM_PROMPT,
        "enable_thinking": False,
        "seed": SEED,
        "n_samples": len(records),
        "n_generations": len(generation_records),
        "reward_score": overall,
        "mra_reward": overall,
        "mra_average_reward": macro,
        **{f"mra_{category}_reward": category_means[category] for category in categories},
        "counts": {category: len(per_category[category]) for category in categories},
        "generation_config": {
            "max_prompt_length": MAX_PROMPT_LENGTH,
            "max_response_length": MAX_RESPONSE_LENGTH,
            "max_model_len": MAX_MODEL_LEN,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": settings["gpu_memory_utilization"],
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 128,
            "enforce_eager": False,
            "enable_chunked_prefill": settings["enable_chunked_prefill"],
            "temperature": SAMPLING_CONFIG["temperature"],
            "top_p": SAMPLING_CONFIG["top_p"],
            "top_k": SAMPLING_CONFIG["top_k"],
            "min_p": SAMPLING_CONFIG["min_p"],
            "presence_penalty": SAMPLING_CONFIG["presence_penalty"],
            "repetition_penalty": SAMPLING_CONFIG["repetition_penalty"],
            "seed": SEED,
            "video_nframes": VIDEO_NFRAMES,
            "video_timestamp_fps": VIDEO_TIMESTAMP_FPS,
            "video_fps": VIDEO_FPS,
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,
            "format_prompt": str(FORMAT_PROMPT.relative_to(REPOSITORY_ROOT)),
            "chat_template": str(CHAT_TEMPLATE.relative_to(REPOSITORY_ROOT)),
        },
        "generation_stats": {
            "finish_reason_counts": dict(sorted(finish_counts.items())),
            "length_capped_count": finish_counts.get("length", 0),
            "missing_think_end_count": 0,
            "generated_tokens_mean": float(np.mean(token_counts)) if token_counts else None,
            "generated_tokens_p95": _percentile(token_counts, 95),
            "generated_tokens_max": max(token_counts) if token_counts else None,
        },
    }

    print(
        f"[eval] overall={overall if overall is not None else float('nan'):.12f}, "
        f"macro={macro if macro is not None else float('nan'):.12f}",
        flush=True,
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    generations_log.parent.mkdir(parents=True, exist_ok=True)
    with generations_log.open("w", encoding="utf-8") as handle:
        for record in generation_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[eval] predictions: {output_csv}", flush=True)
    print(f"[eval] metrics: {out_json}", flush=True)
    return result


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the QuantiPhy evaluation")
    parser.add_argument("model", choices=tuple(MODEL_SETTINGS))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--input-csv", type=Path, default=_env_path("QUANTIPHY_INPUT_CSV"))
    parser.add_argument("--video-dir", type=Path, default=_env_path("QUANTIPHY_VIDEO_DIR"))
    parser.add_argument("--video-ext", default=DEFAULT_VIDEO_EXT)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--generations-log", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.input_csv is None:
        raise SystemExit(
            "pass --input-csv /path/to/quantiphy_validation.csv "
            "or set QUANTIPHY_INPUT_CSV"
        )
    input_csv = args.input_csv.expanduser().resolve()
    if not input_csv.is_file():
        raise SystemExit(f"input CSV not found: {input_csv}")
    model_path = (args.model_path or REPOSITORY_ROOT / "weights" / args.model).expanduser().resolve()
    if not model_path.is_dir():
        raise SystemExit(f"model not found: {model_path}")
    video_dir = args.video_dir.expanduser().resolve() if args.video_dir else None
    output_csv = (
        args.output_csv or REPOSITORY_ROOT / "outputs" / "quantiphy" / f"{args.model}.csv"
    ).expanduser().resolve()
    out_json = (
        args.out_json or output_csv.with_suffix(".json")
    ).expanduser().resolve()
    generations_log = (
        args.generations_log or output_csv.with_suffix(".generations.jsonl")
    ).expanduser().resolve()
    evaluate(
        args.model,
        model_path,
        input_csv,
        video_dir,
        args.video_ext,
        output_csv,
        out_json,
        generations_log,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
