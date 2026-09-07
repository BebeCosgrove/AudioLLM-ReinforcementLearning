from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
from peft import PeftModel
import librosa
import os
import random
import time
import re
import soundfile as sf
from accelerate import Accelerator
from accelerate.utils import gather_object
from collections import defaultdict
from pathlib import Path


BATCH_SIZE = 4

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

    for batch in loader:
        convs = []
        examples = []

        for ex in batch:
            audio_path = ex["audio_url"]

            start = time.time()
            audio = librosa.load(
                audio_path,
                sr=16000,
                mono=True,
            )[0]

            if accelerator.is_main_process:
                print(
                    f"Load audio: {time.time() - start:.2f}s",
                    flush=True,
                )

            choices_text = "\n".join(ex["choice"])
            full_question = (
                f"{ex['question']}\n{choices_text}"
            )

            conv = [
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
                            "type": "text",
                            "text": full_question,
                        },
                        {
                            "type": "audio",
                            "path": audio,
                        },
                    ],
                },
            ]

            convs.append(conv)
            examples.append(ex)

        inputs = processor.apply_chat_template(
            convs,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            processor_kwargs={
                "padding": "max_length",
                "return_tensors": "pt",
            },
        )

        inputs = {
            key: value.to(accelerator.device)
            for key, value in inputs.items()
        }

        inputs["input_features"] = inputs["input_features"].to(torch.bfloat16)

        start = time.time()

        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                output_ids = unwrapped_model.generate(
                    **inputs,
                    max_new_tokens=4,
                    do_sample=False,
                )

        if accelerator.is_main_process:
            print(
                f"Generation: {time.time() - start:.2f}s",
                flush=True,
            )

        response_ids = output_ids[
            :,
            inputs["input_ids"].shape[1]:,
        ]

        preds = processor.tokenizer.batch_decode(
            response_ids,
            skip_special_tokens=True,
        )

        for ex, pred in zip(examples, preds):
            gt = ex["answer"]

            pred_letter = extract_choice_letter(pred)
            gt_letter = extract_choice_letter(gt)

            is_correct = pred_letter == gt_letter

            results.append(
    {
        "id": ex["id"],
        "question": ex["question"],

        # subgroup information
        "question_type": ex.get(
            "question_type",
            "unknown",
        ),
        "subset": infer_subset(ex),

        "answer": gt,
        "prediction": pred,
        "pred_letter": pred_letter,
        "gt_letter": gt_letter,
        "correct": is_correct,
        "audio_url": ex["audio_url"],
    }
)

            if is_correct:
                correct += 1

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

    global_correct = accelerator.gather(local_correct).sum().item()
    global_total = accelerator.gather(local_total).sum().item()

    all_results = gather_object(results)

    overall_accuracy = (
        global_correct / global_total
        if global_total > 0
        else 0.0
    )


    # ==========================================================
    # ACCURACY BY QUESTION TYPE
    # ==========================================================

    question_type_stats = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )

    for result in all_results:

        question_type = result.get(
            "question_type",
            "unknown",
        )

        question_type_stats[
            question_type
        ]["total"] += 1

        if result["correct"]:
            question_type_stats[
                question_type
            ]["correct"] += 1


    accuracy_by_question_type = {}

    for question_type, counts in question_type_stats.items():

        type_correct = counts["correct"]
        type_total = counts["total"]

        accuracy_by_question_type[
            question_type
        ] = {
            "accuracy": (
                type_correct / type_total
                if type_total > 0
                else 0.0
            ),
            "correct": type_correct,
            "total": type_total,
        }


    # ==========================================================
    # ACCURACY BY DATASET SUBSET
    # Bio / Temporal / Complex
    # ==========================================================

    subset_stats = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )

    for result in all_results:

        subset = result["subset"]

        subset_stats[subset]["total"] += 1

        if result["correct"]:
            subset_stats[
                subset
            ]["correct"] += 1


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


    # ==========================================================
    # DOMAIN AVERAGE
    # ==========================================================

    expected_subsets = (
        "Bio",
        "Temporal",
        "Complex",
    )

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


    # ==========================================================
    # SAVE + PRINT RESULTS
    # ==========================================================

    if accelerator.is_main_process:

        output_path = (
            f"dcase_{output_prefix}_results.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                {
                    "overall": {
                        "accuracy": overall_accuracy,
                        "correct": global_correct,
                        "total": global_total,
                    },

                    "domain_average_accuracy":
                        domain_average_accuracy,

                    "accuracy_by_subset":
                        accuracy_by_subset,

                    "accuracy_by_question_type":
                        accuracy_by_question_type,

                    "results":
                        all_results,
                },
                f,
                indent=2,
            )


        # ------------------------------------------------------
        # Overall
        # ------------------------------------------------------

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


        # ------------------------------------------------------
        # Subsets
        # ------------------------------------------------------

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


        # ------------------------------------------------------
        # Question types
        # ------------------------------------------------------

        print(
            f"\n{output_prefix} accuracy by question type:",
            flush=True,
        )

        for question_type in sorted(
            accuracy_by_question_type
        ):

            stats = accuracy_by_question_type[
                question_type
            ]

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
            "domain_average_accuracy":
                domain_average_accuracy,
            "accuracy_by_subset":
                accuracy_by_subset,
            "accuracy_by_question_type":
                accuracy_by_question_type,
        }


def extract_choice_letter(text):
    """Extract the choice letter (A, B, C, D) from a response or answer string."""
    match = re.search(r'\b([A-D])\b', text.strip().upper())
    return match.group(1) if match else None

def infer_subset(example):
    """
    Infer dataset subset from audio filename.

    fold...  -> Temporal
    audio... -> Complex
    anything else -> Bio
    """

    audio_path = str(example.get("audio_url", ""))
    audio_name = Path(audio_path).name.lower()

    if audio_name.startswith("fold"):
        return "Temporal"

    if audio_name.startswith("audio"):
        return "Complex"

    return "Bio"


def run_evaluation():

    accelerator = Accelerator()
    
    #data
    with open("/data/not_backed_up/cosgrv/af3_project/dcase_2025/2025_DCASE_AudioQA/combined_json/dcase_split_validation.json") as f:
        data = json.load(f)

    # # baseline model
    # baseline_loader = DataLoader(
    #     data,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     collate_fn=lambda x: x,
    # )

    # baseline_model = (
    #     AudioFlamingo3ForConditionalGeneration.from_pretrained(
    #         "nvidia/audio-flamingo-3-hf",
    #         torch_dtype=torch.bfloat16,
    #     )
    # )



    # baseline_processor = AutoProcessor.from_pretrained(
    #     "nvidia/audio-flamingo-3-hf"
    # )

    

    # baseline_model, baseline_loader = accelerator.prepare(
    #     baseline_model,
    #     baseline_loader,
    # )

    # unwrapped_model = accelerator.unwrap_model(baseline_model)

    # baseline_accuracy = evaluate(
    #     baseline_model,
    #     unwrapped_model,
    #     baseline_processor,
    #     baseline_loader,
    #     accelerator,
    #     output_prefix="baseline",
    # )

    # accelerator.wait_for_everyone()

    # del unwrapped_model
    # del baseline_model
    # del baseline_loader

    # accelerator.free_memory()
    # torch.cuda.empty_cache()

    # data = data[:500]

    # LONG_AUDIO_SECONDS = 30

    # long_data = []

    # for ex in data:
    #     info = sf.info(ex["audio_url"])
    #     if info.duration >= LONG_AUDIO_SECONDS:
    #         long_data.append(ex)

    # print(f"Original examples: {len(data)}")
    # print(f"Long audio examples (>= {LONG_AUDIO_SECONDS}s): {len(long_data)}")

    # print("Loaded data")
    # print("Loading model")

    #trained model
    mdpo_loader = DataLoader(
        data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda x: x,
    )

    training_model = (
        AudioFlamingo3ForConditionalGeneration.from_pretrained(
            "nvidia/audio-flamingo-3-hf",
            torch_dtype=torch.bfloat16,
        )
    )

    

    training_model = PeftModel.from_pretrained(
        training_model,
        "/data/not_backed_up/cosgrv/"
            "af3_project/mdpo_runs/checkpoint-epoch-5"
    )

    training_model.config.use_cache = True

    mdpo_processor = AutoProcessor.from_pretrained(
        "/data/not_backed_up/cosgrv/"
            "af3_project/mdpo_runs/checkpoint-epoch-5"
    )

    training_model, mdpo_loader = accelerator.prepare(
        training_model,
        mdpo_loader,
    )

    unwrapped_model = accelerator.unwrap_model(training_model)

    mdpo_accuracy = evaluate(
        training_model,
        unwrapped_model,
        mdpo_processor,
        mdpo_loader,
        accelerator,
        output_prefix="mdpo_epoch1",
    )

    if accelerator.is_main_process:
#         print(
#     "Baseline Overall Accuracy:",
#     baseline_accuracy["overall_accuracy"],
# )

#         print(
#             "Baseline Domain Average:",
#             baseline_accuracy["domain_average_accuracy"],
#         )

        print(
            "mDPO Overall Accuracy:",
            mdpo_accuracy["overall_accuracy"],
        )

        print(
            "mDPO Domain Average:",
            mdpo_accuracy["domain_average_accuracy"],
        )
    
    

if __name__ == "__main__":
    run_evaluation()
