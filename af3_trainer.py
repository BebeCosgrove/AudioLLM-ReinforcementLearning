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

        BASE_DIR = "/data/not_backed_up/cosgrv/af3_project/ah_existence"

        for ex in examples:
            audio_path = os.path.join(BASE_DIR, ex["path"])
            perturbed_audio_path = ex["perturbed_path"]  # already absolute

            audio = librosa.load(audio_path, sr=16000, mono=True)[0]
            perturbed_audio = librosa.load(perturbed_audio_path, sr=16000, mono=True)[0]

            chosen_convs.append([
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "Focus on the given audio and answer the following question with exactly one word: yes or no."
                        }
                    ]
                },
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
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": ex["text"]
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
                            "text": "Focus on the given audio and answer the following question with exactly one word: yes or no."
                        }
                    ]
                },
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
                            "text": "Focus on the given audio and answer the following question with exactly one word: yes or no."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": ex["Q"]
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
                            "text": ex["text"]
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
                            "text": "Focus on the given audio and answer the following question with exactly one word: yes or no."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ex["Q"]},
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
                        "text": "Focus on the given audio and answer the following question with exactly one word: yes or no."
                    }
                ]
            },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ex["Q"]},
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
                            "text": ex["text"]
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

        
        chosen_inputs = self.processor.apply_chat_template(
        chosen_convs,
        tokenize=True,
        return_dict=True,
        output_labels=True,
        add_generation_prompt=False
    )

        rejected_inputs = self.processor.apply_chat_template(
        rejected_convs,
        tokenize=True,
        return_dict=True,
        output_labels=True,
        add_generation_prompt=False
    )
    
        perturbed_inputs = self.processor.apply_chat_template(
        perturbed_convs,
        tokenize=True,
        return_dict=True,
        output_labels=True,
        add_generation_prompt=False
    )
        
        prompt_inputs = self.processor.apply_chat_template(
        prompt_convs,
        tokenize=True,
        return_dict=True,
        add_generation_prompt=True
    )
        perturbed_prompt_inputs = self.processor.apply_chat_template(
        perturbed_prompt_convs,
        tokenize=True,
        return_dict=True,
        add_generation_prompt=True
    )
        import pdb; pdb.set_trace()

        print(self.processor.tokenizer.decode(chosen_inputs['input_ids']))

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
        "perturbed": perturbed_inputs
        }


def mdpo_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor, 
    policy_perturbed_chosen_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor, 
    reference_perturbed_chosen_logps: torch.FloatTensor,
    beta = 0.1,
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
    losses = -torch.nn.functional.logsigmoid(beta * logits) \
            -torch.nn.functional.logsigmoid(beta * audio_conditional_logits) \
            -torch.nn.functional.logsigmoid(beta * anchor_logits)

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


def get_sequence_logps(logits, labels):
    #makes the logits and labels aligned from the shift
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    #convert logits -> log probabilities
    log_probs = torch.log_softmax(shift_logits, dim=-1)

    mask = shift_labels != -100 # bool for where tokens are/ aren't -100
    safe_labels = shift_labels.clone()
    safe_labels[~mask] = 0 # replaces places that were -100 with a 0 because gather uses 0

    token_logps = log_probs.gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)

    return (token_logps * mask).sum(dim=-1)


def evaluate(model, processor, dataset):
    correct = 0
    BASE_DIR = "/data/not_backed_up/cosgrv/af3_project/ah_existence"
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



def run():
    device = "cuda"
    beta = 0.1

    wandb.init(
        project="af3-mdpo",
        config={
            "beta": beta,
            "lr": 5e-6,
            "batch_size": 2,
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
    processor = AutoProcessor.from_pretrained(model_id)
    #policy model
    policy_model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    policy_model = get_peft_model(policy_model, lora_config)


    with open("/data/not_backed_up/cosgrv/af3_project/ah_existence/perturbed_datasets/ah_existence_no_audio_train.json") as f:
        data = json.load(f)

    with open("/data/not_backed_up/cosgrv/af3_project/ah_existence/perturbed_datasets/ah_existence_no_audio_val.json") as f:
        val_data = json.load(f)



    #gets train dataset to pytorch form
    train_dataset = AudioDPODataset(data)

    #gets collator with our type of processor
    collator= AudioMDPOCollator(processor)

    # get dataloader
    loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn= collator
    )

    
    #optimizer with 1e-6 learning rate
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-6)

    

    for epoch in range(3):
        #training mode
        policy_model.train()

        for batch in loader:
            print(f"[Start] Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

            #gets the input sequence from that batch
            chosen_inputs = batch["chosen"]
            rejected_inputs = batch["rejected"]
            perturbed_inputs = batch["perturbed"]  

            chosen_inputs["input_features"] = (
            chosen_inputs["input_features"].to(torch.bfloat16)
            )

            rejected_inputs["input_features"] = (
                rejected_inputs["input_features"].to(torch.bfloat16)
            )

            perturbed_inputs["input_features"] = (
                perturbed_inputs["input_features"].to(torch.bfloat16)
            )
                
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

            #forward pass
            policy_chosen_outputs = policy_model(**chosen_inputs)
            policy_rejected_outputs = policy_model(**rejected_inputs)
            policy_perturbed_outputs = policy_model(**perturbed_inputs)

            print(f"[After forward] Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

            #freezes reference model
            with policy_model.disable_adapter():
                with torch.no_grad():
                    reference_chosen_outputs = policy_model(**chosen_inputs)
                    reference_rejected_outputs = policy_model(**rejected_inputs)
                    reference_perturbed_outputs = policy_model(**perturbed_inputs)

            #logps
            policy_chosen_logps = get_sequence_logps(policy_chosen_outputs.logits, chosen_labels)
            policy_rejected_logps = get_sequence_logps(policy_rejected_outputs.logits, rejected_labels)
            policy_perturbed_logps = get_sequence_logps(policy_perturbed_outputs.logits, perturbed_labels)

            reference_chosen_logps = get_sequence_logps(reference_chosen_outputs.logits, chosen_labels)
            reference_rejected_logps = get_sequence_logps(reference_rejected_outputs.logits, rejected_labels)
            reference_perturbed_logps = get_sequence_logps(reference_perturbed_outputs.logits, perturbed_labels)



            # computing mdpo loss
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

            wandb.log({
                "epoch": epoch,
                "loss": loss.item(),
                "chosen_reward": chosen_reward_mean,
                "rejected_reward": rejected_reward_mean,
                "perturbed_reward": perturbed_reward_mean,
                "reward_margin": chosen_reward_mean - rejected_reward_mean,
            })

            # backward + update
            optimizer.zero_grad()
            loss.backward()
            print(f"[After step] Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
            
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

        policy_model.eval()
        

        val_acc = evaluate(
        policy_model,
        processor,
        val_data
        )

        wandb.log({
        "epoch": epoch,
        "val_accuracy": val_acc,
        })
        

    policy_model.save_pretrained("/data/not_backed_up/cosgrv/af3_project/mdpo_runs/checkpoint-final")
    processor.save_pretrained("/data/not_backed_up/cosgrv//af3_project/mdpo_runs/checkpoint-final")

    wandb.finish()



if __name__ == "__main__":
    run()
        


