from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
)
from torch.utils.data import DataLoader
from peft import PeftModel
from accelerate import Accelerator
from accelerate.utils import gather_object
from collections import defaultdict
from pathlib import Path

import torch
import json
import librosa
import time
import re
import os

BATCH_SIZE = 4

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"

LOCAL_MODEL_PATH = "/data/not_backed_up/cosgrv/huggingface_cache/hub/models--Qwen--Qwen2-Audio-7B-Instruct/snapshots/0a095220c30b7b31434169c3086508ef3ea5bf0a"

CACHE_DIR = (
    "/data/not_backed_up/cosgrv/"
    "huggingface_cache/hub"
)

DATA_PATH = (
    "/data/not_backed_up/cosgrv/af3_project/"
    "dcase_2025/2025_DCASE_AudioQA/"
    "combined_json/dcase_split_test.json"
)

QWEN_CHECKPOINT = (
    "/data/not_backed_up/cosgrv/af3_project/mdpo_runs/qwen2/medium_time_mask/checkpoint-epoch-1"
)

def extract_choice_letter(text):
    """Extract A, B, C, or D from an answer string."""

    match = re.search(
        r"\b([A-D])\b",
        text.strip().upper(),
    )

    return match.group(1) if match else None



def infer_subset(example):
    """
    Infer the dataset subset from the beginning of the audio filename.

    Rules:
        fold...  -> Temporal
        audio... -> Complex
        anything else -> Bio

    The extension and directory path do not matter.
    """
    audio_path = str(example.get("audio_url", ""))
    audio_name = Path(audio_path).name.lower()

    if audio_name.startswith("fold"):
        return "Temporal"

    if audio_name.startswith("audio"):
        return "Complex"

    return "Bio"

def evaluate(
    model,
    unwrapped_model,
    processor,
    loader,
    accelerator,
    output_prefix,
):
    model.eval()

    correct = 0
    results = []

    sampling_rate = (
        processor.feature_extractor.sampling_rate
    )

    for batch_index, batch in enumerate(loader):
        conversations = []
        audios = []
        examples = []

        audio_load_start = time.time()

        for example in batch:
            audio_path = example["audio_url"]

            audio = librosa.load(
                audio_path,
                sr=sampling_rate,
                mono=True,
            )[0]

            choices_text = "\n".join(
                example["choice"]
            )

            full_question = (
                f"{example['question']}\n"
                f"{choices_text}\n"
                "Answer with exactly one letter: "
                "A, B, C, or D."
            )

            conversation = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Focus on the given audio and answer the "
                                "following multiple-choice question. "
                                "Respond with the letter of the correct "
                                "answer (A, B, C, or D)."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio",
                            "audio": audio,
                        },
                        {
                            "type": "text",
                            "text": full_question,
                        },
                    ],
                },
            ]

            conversations.append(conversation)
            audios.append(audio)
            examples.append(example)

        if accelerator.is_main_process:
            print(
                f"{output_prefix} batch {batch_index}: "
                f"loaded audio in "
                f"{time.time() - audio_load_start:.2f}s",
                flush=True,
            )

        formatted_texts = processor.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=formatted_texts,
            audio=audios,
            sampling_rate=sampling_rate,
            padding=True,
            return_tensors="pt",
        )

        inputs = {
            key: (
                value.to(accelerator.device)
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in inputs.items()
        }

        generation_start = time.time()

        with torch.inference_mode():
            output_ids = unwrapped_model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
            )

        if accelerator.is_main_process:
            print(
                f"{output_prefix} batch {batch_index}: "
                f"generation took "
                f"{time.time() - generation_start:.2f}s",
                flush=True,
            )

        # Remove prompt tokens, leaving only newly generated tokens.
        prompt_length = inputs["input_ids"].shape[1]
        response_ids = output_ids[:, prompt_length:]

        predictions = processor.batch_decode(
            response_ids,
            skip_special_tokens=True,
        )

        for example, prediction in zip(
            examples,
            predictions,
        ):
            ground_truth = example["answer"]

            predicted_letter = extract_choice_letter(
                prediction
            )
            ground_truth_letter = extract_choice_letter(
                ground_truth
            )

            is_correct = (
                predicted_letter == ground_truth_letter
            )

            results.append(
                {
                    "id": example["id"],
                    "question": example["question"],
                    "question_type": example.get(
                        "question_type",
                        "unknown",
                    ),
                    "subset": infer_subset(example),
                    "answer": ground_truth,
                    "prediction": prediction,
                    "pred_letter": predicted_letter,
                    "gt_letter": ground_truth_letter,
                    "correct": is_correct,
                    "audio_url": example["audio_url"],
                }
            )

            if is_correct:
                correct += 1

    # Sum local counts across all distributed processes.
    local_correct = torch.tensor(
        [correct],
        device=accelerator.device,
        dtype=torch.long,
    )
    local_total = torch.tensor(
        [len(results)],
        device=accelerator.device,
        dtype=torch.long,
    )

    global_correct = (
        accelerator.gather(local_correct)
        .sum()
        .item()
    )
    global_total = (
        accelerator.gather(local_total)
        .sum()
        .item()
    )

    # Gather every prediction dictionary from every process.
    all_results = gather_object(results)

    overall_accuracy = (
        global_correct / global_total
        if global_total > 0
        else 0.0
    )

    # --------------------------------------------------
    # Accuracy by question type
    # --------------------------------------------------
    question_type_stats = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )

    for result in all_results:
        question_type = result.get(
            "question_type",
            "unknown",
        )
        question_type_stats[question_type]["total"] += 1

        if result["correct"]:
            question_type_stats[question_type]["correct"] += 1

    accuracy_by_question_type = {}

    for question_type, counts in question_type_stats.items():
        type_correct = counts["correct"]
        type_total = counts["total"]

        accuracy_by_question_type[question_type] = {
            "accuracy": (
                type_correct / type_total
                if type_total > 0
                else 0.0
            ),
            "correct": type_correct,
            "total": type_total,
        }

    # --------------------------------------------------
    # Accuracy by Bio / Temporal / Complex subset
    # --------------------------------------------------
    subset_stats = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )

    for result in all_results:
        subset = result["subset"]
        subset_stats[subset]["total"] += 1

        if result["correct"]:
            subset_stats[subset]["correct"] += 1

    accuracy_by_subset = {}

    for subset, counts in subset_stats.items():
        subset_correct = counts["correct"]
        subset_total = counts["total"]

        accuracy_by_subset[subset] = {
            "accuracy": (
                subset_correct / subset_total
                if subset_total > 0
                else 0.0
            ),
            "correct": subset_correct,
            "total": subset_total,
        }

    # Macro average gives each of the three subsets equal weight.
    expected_subsets = ("Bio", "Temporal", "Complex")
    available_subset_accuracies = [
        accuracy_by_subset[subset]["accuracy"]
        for subset in expected_subsets
        if subset in accuracy_by_subset
    ]

    domain_average_accuracy = (
        sum(available_subset_accuracies)
        / len(available_subset_accuracies)
        if available_subset_accuracies
        else 0.0
    )

    if accelerator.is_main_process:
        output_path = (
            f"dcase_verylight_time_mask_epoch_1.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "overall": {
                        "accuracy": overall_accuracy,
                        "correct": global_correct,
                        "total": global_total,
                    },
                    "domain_average_accuracy": (
                        domain_average_accuracy
                    ),
                    "accuracy_by_subset": (
                        accuracy_by_subset
                    ),
                    "accuracy_by_question_type": (
                        accuracy_by_question_type
                    ),
                    "results": all_results,
                },
                file,
                indent=2,
            )

        print(
            f"\n{output_prefix} overall accuracy: "
            f"{overall_accuracy:.4f} "
            f"({global_correct}/{global_total})",
            flush=True,
        )

        print(
            f"{output_prefix} domain-average accuracy: "
            f"{domain_average_accuracy:.4f}",
            flush=True,
        )

        print(
            f"\n{output_prefix} accuracy by subset:",
            flush=True,
        )

        for subset in expected_subsets:
            if subset not in accuracy_by_subset:
                print(
                    f"  {subset}: no examples found",
                    flush=True,
                )
                continue

            stats = accuracy_by_subset[subset]

            print(
                f"  {subset}: "
                f"{stats['accuracy']:.4f} "
                f"({stats['correct']}/{stats['total']})",
                flush=True,
            )

        print(
            f"\n{output_prefix} accuracy by question type:",
            flush=True,
        )

        for question_type in sorted(
            accuracy_by_question_type
        ):
            stats = accuracy_by_question_type[question_type]

            print(
                f"  {question_type}: "
                f"{stats['accuracy']:.4f} "
                f"({stats['correct']}/{stats['total']})",
                flush=True,
            )

        print(
            f"\nSaved results to {output_path}",
            flush=True,
        )

    accelerator.wait_for_everyone()

    return {
        "overall_accuracy": overall_accuracy,
        "domain_average_accuracy": domain_average_accuracy,
        "accuracy_by_subset": accuracy_by_subset,
        "accuracy_by_question_type": (
            accuracy_by_question_type
        ),
    }

def create_loader(data):
    return DataLoader(
        data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda examples: examples,
        num_workers=0,
    )


def run_evaluation():
    accelerator = Accelerator(
        mixed_precision="bf16",
    )

    with open(
        DATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    accelerator.print(
        f"Loaded {len(data)} test examples"
    )

    # --------------------------------------------------
    # Baseline Qwen2-Audio model
    # --------------------------------------------------

    # baseline_loader = create_loader(data)

    # baseline_processor = (
    #     AutoProcessor.from_pretrained(
    #         LOCAL_MODEL_PATH,
    #         cache_dir=CACHE_DIR,
    #         trust_remote_code=True,
    #     )
    # )

    # baseline_model = (
    #     Qwen2AudioForConditionalGeneration
    #     .from_pretrained(
    #         LOCAL_MODEL_PATH,
    #         torch_dtype=torch.bfloat16,
    #         cache_dir=CACHE_DIR,
    #         trust_remote_code=True,
    #     )
    # )

    # baseline_model.config.use_cache = True

    # baseline_model, baseline_loader = (
    #     accelerator.prepare(
    #         baseline_model,
    #         baseline_loader,
    #     )
    # )

    # baseline_unwrapped_model = (
    #     accelerator.unwrap_model(
    #         baseline_model
    #     )
    # )

    # accelerator.print(
    #     "Starting baseline evaluation"
    # )

    # baseline_accuracy = evaluate(
    #     model=baseline_model,
    #     unwrapped_model=baseline_unwrapped_model,
    #     processor=baseline_processor,
    #     loader=baseline_loader,
    #     accelerator=accelerator,
    #     output_prefix="baseline",
    # )

    # # Free the baseline before loading another full model.
    # accelerator.wait_for_everyone()

    # del baseline_unwrapped_model
    # del baseline_model
    # del baseline_loader
    # del baseline_processor

    # accelerator.free_memory()
    # torch.cuda.empty_cache()

    # accelerator.wait_for_everyone()

    # --------------------------------------------------
    # Qwen2-Audio mDPO model
    # --------------------------------------------------

    mdpo_loader = create_loader(data)

    
    training_base_model = (
        Qwen2AudioForConditionalGeneration
        .from_pretrained(
            LOCAL_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        )
    )

    training_model = PeftModel.from_pretrained(
        training_base_model,
        QWEN_CHECKPOINT,
    )

    training_model.config.use_cache = True

    print("Adapter loaded successfully")
    print(training_model.peft_config["default"].target_modules)

    # Prefer the processor saved with the checkpoint.
    if os.path.exists(QWEN_CHECKPOINT):
        mdpo_processor = (
            AutoProcessor.from_pretrained(
                QWEN_CHECKPOINT,
                cache_dir=CACHE_DIR,
                trust_remote_code=True,
            )
        )
    else:
        mdpo_processor = (
            AutoProcessor.from_pretrained(
                MODEL_ID,
                cache_dir=CACHE_DIR,
                trust_remote_code=True,
            )
        )

    training_model, mdpo_loader = (
        accelerator.prepare(
            training_model,
            mdpo_loader,
        )
    )

    training_unwrapped_model = (
        accelerator.unwrap_model(
            training_model
        )
    )

    accelerator.print(
        "Starting mDPO evaluation"
    )

    mdpo_accuracy = evaluate(
        model=training_model,
        unwrapped_model=training_unwrapped_model,
        processor=mdpo_processor,
        loader=mdpo_loader,
        accelerator=accelerator,
        output_prefix="mdpo_epoch1",
    )

    if accelerator.is_main_process:
        print(
            "\nFinal results",
            flush=True,
        )

        # print(
        #     f"Baseline overall accuracy: "
        #     f"{baseline_accuracy['overall_accuracy']:.4f}",
        #     flush=True,
        # )

        print(
            f"mDPO overall accuracy: "
            f"{mdpo_accuracy['overall_accuracy']:.4f}",
            flush=True,
        )

        # print(
        #     f"Baseline domain-average accuracy: "
        #     f"{baseline_accuracy['domain_average_accuracy']:.4f}",
        #     flush=True,
        # )

        print(
            f"mDPO domain-average accuracy: "
            f"{mdpo_accuracy['domain_average_accuracy']:.4f}",
            flush=True,
        )

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    run_evaluation()