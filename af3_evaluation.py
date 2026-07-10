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


BATCH_SIZE = 4

def evaluate(model, processor, dataset):
    model.eval()
    correct = 0
    results = []

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda x: x,   # keep batch as list of dicts
    )

    for batch in loader:

        convs = []
        examples = []

        for ex in batch:

            audio_path = ex["audio_url"]

            start = time.time()

            audio = librosa.load(audio_path, sr=16000, mono=True)[0]

            print(f"Load audio: {time.time() - start:.2f}s")

            choices_text = "\n".join(ex["choice"])
            full_question = f"{ex['question']}\n{choices_text}"

            conv = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "Focus on the given audio and answer the following multiple-choice question. Respond with only the letter of the correct answer (A, B, C, or D)."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": full_question
                        },
                        {
                            "type": "audio",
                            "path": audio
                        }
                    ]
                }
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
    

        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        inputs["input_features"] = inputs["input_features"].to(torch.bfloat16)

        start = time.time()
    


        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False
            )

        print(f"Generation: {time.time() - start:.2f}s")

        response_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        preds = processor.tokenizer.batch_decode(
            response_ids,
            skip_special_tokens=True
        )

        for ex, pred in zip(examples, preds):

            gt = ex["answer"]

            pred_letter = extract_choice_letter(pred)
            gt_letter = extract_choice_letter(gt)

            results.append({
                "id": ex["id"],
                "question": ex["question"],
                "answer": gt,
                "prediction": pred,
                "pred_letter": pred_letter,
                "gt_letter": gt_letter,
                "correct": pred_letter == gt_letter,
                "audio_url": ex["audio_url"],
            })

            if pred_letter == gt_letter:
                correct += 1

    with open("dcase_mdpo_epoch1_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return correct / len(dataset)

def extract_choice_letter(text):
    """Extract the choice letter (A, B, C, D) from a response or answer string."""
    match = re.search(r'\b([A-D])\b', text.strip().upper())
    return match.group(1) if match else None


def run_evaluation():
    
    #data
    with open("/data/not_backed_up/cosgrv/af3_project/dcase_2025/2025_DCASE_AudioQA/combined_json/dcase_split_test.json") as f:
        data = json.load(f)

    data = data[:500]

    # LONG_AUDIO_SECONDS = 30

    # long_data = []

    # for ex in data:
    #     info = sf.info(ex["audio_url"])
    #     if info.duration >= LONG_AUDIO_SECONDS:
    #         long_data.append(ex)

    # print(f"Original examples: {len(data)}")
    # print(f"Long audio examples (>= {LONG_AUDIO_SECONDS}s): {len(long_data)}")

    print("Loaded data")
    print("Loading model")



    baseline_model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            "nvidia/audio-flamingo-3-hf",
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
    
    baseline_processor = AutoProcessor.from_pretrained("nvidia/audio-flamingo-3-hf")

    print("Loading model")

    training_model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
    "nvidia/audio-flamingo-3-hf",
    device_map="auto",
    torch_dtype=torch.bfloat16
    )

    #Apply trained LoRA adapter
    training_model = PeftModel.from_pretrained(
    training_model,
    "/data/not_backed_up/cosgrv/af3_project/mdpo_runs/checkpoint-epoch-1"
    )

    training_model.config.use_cache = True

    mdpo_processor = AutoProcessor.from_pretrained("/data/not_backed_up/cosgrv/af3_project/mdpo_runs/checkpoint-epoch-1")
    # print(mdpo_processor.feature_extractor)
    # print(mdpo_processor.feature_extractor.__dict__)
    

    
    print("Model loaded")
    print("Starting evaluation")    

    mdpo_accuracy = evaluate(training_model, mdpo_processor, data)
    baseline_accuracy = evaluate(baseline_model, baseline_processor, data)

    print("Baseline Accuracy:", baseline_accuracy)
    print("mDPO Accuracy:", mdpo_accuracy)
    

if __name__ == "__main__":
    run_evaluation()
