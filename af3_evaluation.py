from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
import peft
from peft import PeftModel
import librosa
import os
import random

BASE_DIR = "/data/not_backed_up/cosgrv/af3_project/ah_existence"

def evaluate(model, processor, dataset):
    correct = 0

    for ex in dataset:
        
        audio_path = os.path.join(BASE_DIR, ex["path"])
            
        audio = librosa.load(audio_path, sr=16000, mono=True)[0]
    
        conv = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": ex["Q"]
                    },
                    {
                        "type": "audio",
                        "path": audio
                    }
                ]
            }
        ]

        inputs = processor.apply_chat_template(
            conv,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Cast input_features to match model dtype
        inputs["input_features"] = inputs["input_features"].to(torch.bfloat16)

        output_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)

        # pred = processor.tokenizer.decode(
        #     output_ids[0],
        #     skip_special_tokens=True
        # )

        gt = ex["text"]

        response_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        pred = processor.tokenizer.decode(response_ids[0], skip_special_tokens=True)

        if pred.strip().lower() == gt.strip().lower():
            correct += 1

    return correct / len(dataset)


def run_evaluation():

    #data
    with open("/data/not_backed_up/cosgrv/af3_project/ah_existence/perturbed_datasets/ah_existence_no_audio_test.json") as f:
        data = json.load(f)



    baseline_model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            "nvidia/audio-flamingo-3-hf",
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
    
    baseline_processor = AutoProcessor.from_pretrained("nvidia/audio-flamingo-3-hf")


    training_model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
    "nvidia/audio-flamingo-3-hf",
    device_map="auto",
    torch_dtype=torch.bfloat16
    )

    # Apply trained LoRA adapter
    training_model = PeftModel.from_pretrained(
    training_model,
    "/data/not_backed_up/cosgrv/af3_project/mdpo_runs/checkpoint-final"
    )

    mdpo_processor = AutoProcessor.from_pretrained("/data/not_backed_up/cosgrv/af3_project/mdpo_runs/checkpoint-final")
    

    baseline_accuracy = evaluate(baseline_model, baseline_processor, data)

    mdpo_accuracy = evaluate(training_model, mdpo_processor, data)

    print("Baseline Accuracy:", baseline_accuracy)
    print("mDPO Accuracy:", mdpo_accuracy)
    

if __name__ == "__main__":
    run_evaluation()
