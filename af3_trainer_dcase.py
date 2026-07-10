import sys
from pathlib import Path

# helpers/profiling.py lives under ah_existence/helpers/ — add that dir to sys.path so
# `from helpers.profiling import ...` resolves regardless of this script's cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent / "ah_existence"))

from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
import librosa
import os
import peft
from peft import LoraConfig, get_peft_model
import wandb
import re

from helpers.profiling import Profiler, get_profile_report_path

# Shared profiler instance — disabled (true no-op) unless run() is called with profile=True.
# Reassigned in run(); referenced directly as a module global from AudioMDPOCollator.__call__
# and evaluate() so no signature changes are needed there (same pattern as run_af3.py).
PROFILER = Profiler(enabled=False)


#make dataset into pytorch dataset
class AudioDPODataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class AudioMDPOCollator:

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):

        chosen_convs = []
        rejected_convs = []
        perturbed_convs = []
        prompt_convs = []
        perturbed_prompt_convs = []
        chosen_response_convs = []
        rejected_response_convs = []

        for ex in examples:
            audio_path = ex["audio_url"]
            perturbed_audio_path = ex["perturbed_path"]  # already absolute

            # MAX_AUDIO_SECONDS = 10
            # MAX_AUDIO_SAMPLES = 16000 * MAX_AUDIO_SECONDS

            with PROFILER.section("collator_audio_load", track_key=str(audio_path)):
                audio = librosa.load(audio_path, sr=16000, mono=True)[0]
            #audio = audio[:MAX_AUDIO_SAMPLES]

            with PROFILER.section("collator_audio_load", track_key=str(perturbed_audio_path)):
                perturbed_audio = librosa.load(perturbed_audio_path, sr=16000, mono=True)[0]
            #perturbed_audio = perturbed_audio[:MAX_AUDIO_SAMPLES]

            choices_text = "\n".join(ex["choice"])
            full_question = f"{ex['question']}\n{choices_text}"

            chosen_convs.append([
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
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": ex["answer"]
                        }
                    ]
                }
            ])

            rejected_convs.append([
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
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": ex["rejected"]
                        }
                    ]
                }
            ])

            perturbed_convs.append([
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
                            "path": perturbed_audio
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": ex["answer"]
                        }
                    ]
                }
            ])

            prompt_convs.append([
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
                        {"type": "text", "text": full_question},
                        {"type": "audio", "path": audio}
                    ]
                }
            ])

            perturbed_prompt_convs.append([

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
                        {"type": "text", "text": full_question},
                        {
                            "type": "audio",
                            "path": perturbed_audio
                        }
                    ]
                }
            ])

            chosen_response_convs.append([

                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": ex["answer"]
                        }
                    ]
                }
            ])

            rejected_response_convs.append([

                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": ex["rejected"]
                        }
                    ]
                }
            ])


        with PROFILER.section("collate_encode_chosen", batch_size=len(examples)):
            chosen_inputs = self.processor.apply_chat_template(
            chosen_convs,
            tokenize=True,
            return_dict=True,
            output_labels=True,
            add_generation_prompt=False
        )

        with PROFILER.section("collate_encode_rejected", batch_size=len(examples)):
            rejected_inputs = self.processor.apply_chat_template(
            rejected_convs,
            tokenize=True,
            return_dict=True,
            output_labels=True,
            add_generation_prompt=False
        )

        with PROFILER.section("collate_encode_perturbed", batch_size=len(examples)):
            perturbed_inputs = self.processor.apply_chat_template(
            perturbed_convs,
            tokenize=True,
            return_dict=True,
            output_labels=True,
            add_generation_prompt=False
        )

        with PROFILER.section("collate_encode_prompt", batch_size=len(examples)):
            prompt_inputs = self.processor.apply_chat_template(
            prompt_convs,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True
        )
        with PROFILER.section("collate_encode_perturbed_prompt", batch_size=len(examples)):
            perturbed_prompt_inputs = self.processor.apply_chat_template(
            perturbed_prompt_convs,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True
        )
        #import pdb; pdb.set_trace()


        with PROFILER.section("collate_build_labels", batch_size=len(examples)):
            #gets the length of the prompt
            prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1)
            perturbed_prompt_lengths = (perturbed_prompt_inputs["attention_mask"].sum(dim=1)

            )

            #makes a copy of chosen_inputs where everything is -100
            chosen_labels = torch.full_like(
            chosen_inputs["input_ids"],
            -100
            )
            rejected_labels = torch.full_like(
            rejected_inputs["input_ids"],
            -100
            )
            perturbed_labels = torch.full_like(
            perturbed_inputs["input_ids"],
            -100
            )


            #makes everything after response have their actual input_ids instead of -100
            for b in range(len(examples)):
                chosen_mask = chosen_inputs["attention_mask"][b]
                chosen_start = chosen_mask.nonzero()[0].item()
                prompt_len = prompt_lengths[b].item()
                response_start = chosen_start + prompt_len
                chosen_labels[b, response_start:] = chosen_inputs["input_ids"][b, response_start:]
            for b in range(len(examples)):
                rejected_mask = rejected_inputs["attention_mask"][b]
                rejected_start = rejected_mask.nonzero()[0].item()
                prompt_len = prompt_lengths[b].item()
                response_start = rejected_start + prompt_len
                rejected_labels[b, response_start:] = rejected_inputs["input_ids"][b, response_start:]

            for b in range(len(examples)):
                perturbed_mask = perturbed_inputs["attention_mask"][b]
                perturbed_start = perturbed_mask.nonzero()[0].item()
                perturbed_prompt_len = perturbed_prompt_lengths[b].item()
                response_start = perturbed_start + perturbed_prompt_len
                perturbed_labels[b, response_start:] = perturbed_inputs["input_ids"][b, response_start:]



            #assigns the labels to the new fixed labels that only have ids for the response
            chosen_inputs["labels"] = chosen_labels
            rejected_inputs["labels"] = rejected_labels
            perturbed_inputs["labels"] = perturbed_labels


        return {
    "chosen": chosen_inputs,
    "rejected": rejected_inputs,
    "perturbed": perturbed_inputs,
    "ids": [ex["id"] for ex in examples],
    "audio_urls": [ex["audio_url"] for ex in examples],
}


def mdpo_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    policy_perturbed_chosen_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    reference_perturbed_chosen_logps: torch.FloatTensor,
    beta = 0.05,
    reference_free: bool = False):

    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # response preference

    audio_conditional_pi_logratios = policy_chosen_logps - policy_perturbed_chosen_logps
    audio_conditional_ref_logratios = reference_chosen_logps - reference_perturbed_chosen_logps

    if reference_free:
        audio_conditional_ref_logratios = 0

    audio_conditional_logits = audio_conditional_pi_logratios - audio_conditional_ref_logratios  # audio-conditional preference

    anchor_logits = policy_chosen_logps - reference_chosen_logps  # anchored preference

    # mDPO
    losses = -torch.nn.functional.logsigmoid(beta * logits)\
        -torch.nn.functional.logsigmoid(beta * anchor_logits)
    -torch.nn.functional.logsigmoid(beta * audio_conditional_logits)



    chosen_rewards = (
        beta * (policy_chosen_logps - reference_chosen_logps).detach()
        )
    rejected_rewards = (
        beta * (policy_rejected_logps - reference_rejected_logps).detach()
        )
    perturbed_rewards = (
        beta * (policy_perturbed_chosen_logps - reference_perturbed_chosen_logps).detach()
        )

    return losses, chosen_rewards, rejected_rewards, perturbed_rewards


# def get_sequence_logps(logits, labels):
#     #makes the logits and labels aligned from the shift
#     shift_logits = logits[:, :-1, :]
#     shift_labels = labels[:, 1:]

#     #convert logits -> log probabilities
#     log_probs = torch.log_softmax(shift_logits.float(), dim=-1)

#     mask = shift_labels != -100 # bool for where tokens are/ aren't -100
#     safe_labels = shift_labels.clone()
#     safe_labels[~mask] = 0 # replaces places that were -100 with a 0 because gather uses 0

#     token_logps = log_probs.gather(
#         dim=-1,
#         index=safe_labels.unsqueeze(-1)
#     ).squeeze(-1)

#     return (token_logps * mask).sum(dim=-1)

def get_sequence_logps(logits, labels):
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]

    log_probs = torch.log_softmax(shift_logits, dim=-1)

    mask = shift_labels != -100
    safe_labels = shift_labels.clone()
    safe_labels[~mask] = 0

    token_logps = log_probs.gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)

    token_counts = mask.sum(dim=-1).clamp(min=1)
    seq_logps = (token_logps * mask).sum(dim=-1)



    if torch.isnan(seq_logps).any() or torch.isinf(seq_logps).any():
        print("NaN in logits:", torch.isnan(logits).any().item())
        print("Inf in logits:", torch.isinf(logits).any().item())
        print("labels valid counts:", (labels[:, 1:] != -100).sum(dim=1))
        raise RuntimeError("NaN/Inf in sequence logps")

    return seq_logps


def evaluate(model, processor, dataset, max_samples=None):
    correct = 0
    #results = []

    samples = dataset if max_samples is None else dataset[:max_samples]

    for ex in samples:

        with PROFILER.section("eval_sample_total"):
            audio_path = ex["audio_url"]

            with PROFILER.section("eval_audio_load", track_key=str(audio_path)):
                audio = librosa.load(audio_path, sr=16000, mono=True)[0]

            # Build the question with choices included
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

            with PROFILER.section("eval_encode"):
                inputs = processor.apply_chat_template(
                    conv,
                    tokenize=True,
                    return_dict=True,
                    add_generation_prompt=True
                )

            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Cast input_features to match model dtype
            #inputs["input_features"] = inputs["input_features"].to(torch.bfloat16)

            with PROFILER.section("eval_generate"):
                output_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)

            # pred = processor.tokenizer.decode(
            #     output_ids[0],
            #     skip_special_tokens=True
            # )

            gt = ex["answer"]

            with PROFILER.section("eval_decode"):
                response_ids = output_ids[:, inputs["input_ids"].shape[1]:]
                pred = processor.tokenizer.decode(response_ids[0], skip_special_tokens=True)

                pred_letter = extract_choice_letter(pred)
                gt_letter = extract_choice_letter(gt)

            # results.append({
            #     "id": ex["id"],
            #     "question": ex["question"],
            #     "answer": gt,
            #     "prediction": pred,
            #     "pred_letter": pred_letter,
            #     "gt_letter": gt_letter,
            #     "correct": pred_letter == gt_letter,
            #     "audio_url": audio_path,
            # })

            if pred_letter == gt_letter:
                correct += 1

    # with open("dcase_baseline_results.json", "w") as f:
    #     json.dump(results, f, indent=2)

    return correct / len(samples)

def extract_choice_letter(text):
    """Extract the choice letter (A, B, C, D) from a response or answer string."""
    match = re.search(r'\b([A-D])\b', text.strip().upper())
    return match.group(1) if match else None



def run(profile=False, profile_max_batches=None):
    global PROFILER
    device = "cuda"
    beta = 0.1

    PROFILER = Profiler(enabled=profile)
    if profile:
        print(f"Profiling: enabled (max_batches={profile_max_batches})")
        PROFILER.set_meta(
            mode="dcase_train",
            beta=beta,
            batch_size=6,
            model_id="nvidia/audio-flamingo-3-hf",
        )

    # Profiling runs are diagnostic — don't create wandb experiment entries for them.
    if not profile:
        wandb.init(
            project="af3-mdpo",
            config={
                "beta": beta,
                "lr": 1e-6,
                "batch_size": 6,
                "epochs": 2,
                "lora_r": 8,
                "lora_alpha": 16,
            }
        )

    lora_config = LoraConfig(
    r=8,                    # rank — smaller = fewer params, less expressive
    lora_alpha=16,          # scaling factor
    target_modules=["q_proj", "v_proj"],  # which layers to apply LoRA to
    lora_dropout=0.05,
    bias="none"
)



    model_id = "nvidia/audio-flamingo-3-hf"
    with PROFILER.section("processor_load"):
        processor = AutoProcessor.from_pretrained(model_id)
        processor.max_audio_len = 600
    #policy model
    with PROFILER.section("model_load"):
        policy_model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float32
    )

    policy_model = get_peft_model(policy_model, lora_config)

    policy_model.config.use_cache = False
    policy_model.gradient_checkpointing_enable()


    policy_model.audio_tower.float()
    policy_model.multi_modal_projector.float()

    if profile:
        PROFILER.set_meta(
            device_map=getattr(policy_model, "hf_device_map", None),
            gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
        )
        PROFILER.snapshot_memory("after_model_load")

    CHECKPOINT_DIR = "/data/not_backed_up/cosgrv/af3_project/mdpo_runs"

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


    with open("/data/not_backed_up/cosgrv/af3_project/dcase_2025/2025_DCASE_AudioQA/perturbed_datasets/dcase_train_no_audio_final.json") as f:
        data = json.load(f)


    with open("/data/not_backed_up/cosgrv/af3_project/dcase_2025/2025_DCASE_AudioQA/perturbed_datasets/dcase_val_no_audio_final.json") as f:
        val_data = json.load(f)



    #gets train dataset to pytorch form
    train_dataset = AudioDPODataset(data)

    #gets collator with our type of processor
    collator= AudioMDPOCollator(processor)

    # get dataloader
    loader = DataLoader(
        train_dataset,
        batch_size=6,
        shuffle=True,
        collate_fn= collator
    )

    print("Dataset size:", len(train_dataset))
    print("Number of batches:", len(loader))

    if profile:
        PROFILER.set_meta(num_batches_total=len(loader) * 3)  # 3 epochs


    #optimizer with 1e-6 learning rate
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-6)

    global_step = 0
    profiled_batches = 0
    truncated = False
    for epoch in range(3):
        #training mode
        policy_model.train()

        for batch in loader:
            print(f"[Start] Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

            with PROFILER.section("batch_total"):
                #gets the input sequence from that batch
                chosen_inputs = batch["chosen"]
                rejected_inputs = batch["rejected"]
                perturbed_inputs = batch["perturbed"]

                # chosen_inputs["input_features"] = chosen_inputs["input_features"].float()
                # rejected_inputs["input_features"] = rejected_inputs["input_features"].float()
                # perturbed_inputs["input_features"] = perturbed_inputs["input_features"].float()

                #pops labels and moves the tensors to gpu
                chosen_labels = chosen_inputs.pop("labels")
                rejected_labels = rejected_inputs.pop("labels")
                perturbed_labels = perturbed_inputs.pop("labels")

                #moves to gpu
                chosen_inputs = {k: v.to(policy_model.device) if isinstance(v, torch.Tensor) else v
                                 for k, v in chosen_inputs.items()}
                rejected_inputs = {k: v.to(policy_model.device) if isinstance(v, torch.Tensor) else v
                                   for k, v in rejected_inputs.items()}
                perturbed_inputs = {k: v.to(policy_model.device) if isinstance(v, torch.Tensor) else v
                                    for k, v in perturbed_inputs.items()}

                #moves to gpu
                chosen_labels = chosen_labels.to(policy_model.device)
                rejected_labels = rejected_labels.to(policy_model.device)
                perturbed_labels = perturbed_labels.to(policy_model.device)
                print("Batch ids:", batch["ids"])
                print("Batch audio:", batch["audio_urls"])

                #forward pass
                with PROFILER.section("policy_forward_chosen"):
                    policy_chosen_outputs = policy_model(**chosen_inputs)
                with PROFILER.section("policy_forward_rejected"):
                    policy_rejected_outputs = policy_model(**rejected_inputs)
                with PROFILER.section("policy_forward_perturbed"):
                    policy_perturbed_outputs = policy_model(**perturbed_inputs)

                print(f"[After forward] Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

                #freezes reference model

                with policy_model.disable_adapter():
                    with torch.no_grad():
                        with PROFILER.section("reference_forward_chosen"):
                            reference_chosen_outputs = policy_model(**chosen_inputs)
                        with PROFILER.section("reference_forward_rejected"):
                            reference_rejected_outputs = policy_model(**rejected_inputs)
                        with PROFILER.section("reference_forward_perturbed"):
                            reference_perturbed_outputs = policy_model(**perturbed_inputs)

                print("NaN:", torch.isnan(policy_chosen_outputs.logits).any().item())

                #logps
                with PROFILER.section("compute_logps"):
                    policy_chosen_logps = get_sequence_logps(policy_chosen_outputs.logits, chosen_labels)
                    policy_rejected_logps = get_sequence_logps(policy_rejected_outputs.logits, rejected_labels)
                    policy_perturbed_logps = get_sequence_logps(policy_perturbed_outputs.logits, perturbed_labels)

                    reference_chosen_logps = get_sequence_logps(reference_chosen_outputs.logits, chosen_labels)
                    reference_rejected_logps = get_sequence_logps(reference_rejected_outputs.logits, rejected_labels)
                    reference_perturbed_logps = get_sequence_logps(reference_perturbed_outputs.logits, perturbed_labels)



                # computing mdpo loss
                with PROFILER.section("compute_loss"):
                    losses, chosen_rewards, rejected_rewards, perturbed_rewards = mdpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    policy_perturbed_logps,
                    reference_chosen_logps,
                    reference_rejected_logps,
                    reference_perturbed_logps
                    )

                    #gets average of the batches losses
                    loss = losses.mean()
                    chosen_reward_mean = chosen_rewards.mean().item()
                    rejected_reward_mean = rejected_rewards.mean().item()
                    perturbed_reward_mean = perturbed_rewards.mean().item()

                print(f"Epoch {epoch} | Loss: {loss.item():.4f} | "
                    f"Chosen reward: {chosen_reward_mean:.4f} | "
                    f"Rejected reward: {rejected_reward_mean:.4f} | "
                    f"Perturbed reward: {perturbed_reward_mean:.4f}")
                print(f"[After loss] Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

                if not profile:
                    wandb.log({
                    "train/step": global_step,
                    "train/loss": loss.item(),
                    "train/chosen_reward": chosen_reward_mean,
                    "train/rejected_reward": rejected_reward_mean,
                    "train/perturbed_reward": perturbed_reward_mean,
                    "train/reward_margin": chosen_reward_mean - rejected_reward_mean,
                    }, step=global_step)

                global_step += 1

                # backward + update
                with PROFILER.section("backward"):
                    optimizer.zero_grad()
                    loss.backward()
                print(f"[After step] Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")


                #torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
                with PROFILER.section("optimizer_step"):
                    optimizer.step()

                del policy_chosen_outputs, policy_rejected_outputs, policy_perturbed_outputs
                del reference_chosen_outputs, reference_rejected_outputs, reference_perturbed_outputs
                del losses, chosen_rewards, rejected_rewards, perturbed_rewards, loss
                del policy_chosen_logps, policy_rejected_logps, policy_perturbed_logps
                del reference_chosen_logps, reference_rejected_logps, reference_perturbed_logps
                del chosen_labels, rejected_labels, perturbed_labels
                del chosen_inputs, rejected_inputs, perturbed_inputs
                torch.cuda.empty_cache()

                print(f"[After cleanup] Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

            if profile:
                profiled_batches += 1
                PROFILER.snapshot_memory(f"after_batch_{profiled_batches}")
                if profile_max_batches is not None and profiled_batches >= profile_max_batches:
                    truncated = True
                    break

        if truncated:
            break

        policy_model.eval()

        # Skip end-of-epoch eval/checkpoint entirely for a truncated diagnostic run —
        # evaluate() is itself slow (unbatched) and a partial epoch's checkpoint isn't useful.
        if not (profile and profile_max_batches is not None):

            val_acc = evaluate(
            policy_model,
            processor,
            val_data
            )

            if not profile:
                wandb.log({
                    "val/accuracy": val_acc,
                    "epoch": epoch,
                }, step=global_step)

            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint-epoch-{epoch+1}")

            checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            f"checkpoint-epoch-{epoch+1}"
        )

            policy_model.save_pretrained(checkpoint_path)
            processor.save_pretrained(checkpoint_path)

            print(f"Saved checkpoint to {checkpoint_path}")


    if profile:
        PROFILER.set_meta(num_batches_profiled=profiled_batches, truncated=truncated)
        PROFILER.write_report(get_profile_report_path("dcase_train", tag=f"mdpo_beta_{beta}"))

    if truncated:
        print(f"\n{'='*60}")
        print(f"PROFILING RUN COMPLETE — partial training NOT saved (--profile-max-batches={profile_max_batches}).")
        print(f"Profiled {profiled_batches} batch(es). See the time-usage report above for the breakdown.")
        print(f"{'='*60}")
        return

    policy_model.save_pretrained("/data/not_backed_up/cosgrv/af3_project/mdpo_runs/checkpoint-final")
    processor.save_pretrained("/data/not_backed_up/cosgrv//af3_project/mdpo_runs/checkpoint-final")

    if not profile:
        wandb.finish()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AF3 mDPO trainer (DCASE 2025)")
    parser.add_argument("--profile", action="store_true",
                        help="Enable time/memory profiling (records the full detailed report); "
                             "writes a report under <repo root>/profiler_reports/. "
                             "Skips wandb logging for the run.")
    parser.add_argument("--profile-max-batches", "--profile_max_batches", type=int, default=None,
                        help="Stop after N training batches when profiling (fast diagnostic run — "
                             "nothing is checkpointed in this mode, end-of-epoch eval is skipped)")
    args = parser.parse_args()

    run(profile=args.profile, profile_max_batches=args.profile_max_batches)
