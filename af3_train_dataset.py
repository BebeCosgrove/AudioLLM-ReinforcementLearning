from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


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

        for ex in examples:

            chosen_convs.append([
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe what is happening in the audio."
                        },
                        {
                            "type": "audio",
                            "path": ex["audio"]
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": ex["chosen"]
                        }
                    ]
                }
            ])

            rejected_convs.append([
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe what is happening in the audio."
                        },
                        {
                            "type": "audio",
                            "path": ex["audio"]
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
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe what is happening in the audio."
                        },
                        {
                            "type": "audio",
                            "path": ex["perturbed_audio"]
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": ex["chosen"]
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
    #convert logits -> log probabilities
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    mask = labels != -100
    safe_labels = labels.clone()
    safe_labels[~mask] = 0 

    token_logps = log_probs.gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)


    lengths = mask.sum(dim=-1)

    return (token_logps * mask).sum(dim=-1)


def run():
    device = "cuda"
    beta = 0.1

    
    model_id = "nvidia/audio-flamingo-3-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    #policy model
    policy_model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    #reference model
    reference_model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
        
    )

    #example dataset

    dataset = [
        {
            "audio": "/data/not_backed_up/cosgrv/af3_project/data/3-146965-A-5.wav",
            "perturbed_audio": "/data/not_backed_up/cosgrv/af3_project/debug/perturbation_samples/test/test_pert.wav",
            "chosen": "A cat is meowing.",
            "rejected": "A train is running."
        },
        {
            "audio": "/data/not_backed_up/cosgrv/af3_project/data/_UvwGWvKmcg_1.wav",
            "perturbed_audio": "/data/not_backed_up/cosgrv/af3_project/debug/perturbation_samples/test/test_pert1.wav",
            "chosen": "A car is revving its engine",
            "rejected": "A person is typing."
        },
    ]

    #gets train dataset to pytorch form
    train_dataset = AudioDPODataset(dataset)

    #gets collator with our type of processor
    collator= AudioMDPOCollator(processor)

    # get dataloader
    loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn= collator
    )

    
    #optimizer with 1e-6 learning rate
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-6)

    #evaluates reference and trains the policy
    reference_model.eval()
    policy_model.train()
    

    for epoch in range(10):
        for batch in loader:

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

            labels_chosen = chosen_inputs["labels"]
            labels_rejected = rejected_inputs["labels"]
            labels_perturbed = perturbed_inputs["labels"]



            #forward pass
            policy_chosen_outputs = policy_model(**chosen_inputs)
            policy_rejected_outputs = policy_model(**rejected_inputs)
            policy_perturbed_outputs = policy_model(**perturbed_inputs)

            #freezes reference model
            with torch.no_grad():
                reference_chosen_outputs = reference_model(**chosen_inputs)
                reference_rejected_outputs = reference_model(**rejected_inputs)
                reference_perturbed_outputs = reference_model(**perturbed_inputs)

            #logps
            policy_chosen_logps = get_sequence_logps(policy_chosen_outputs.logits, labels_chosen)
            policy_rejected_logps = get_sequence_logps(policy_rejected_outputs.logits, labels_rejected)
            policy_perturbed_logps = get_sequence_logps(policy_perturbed_outputs.logits, labels_perturbed)

            reference_chosen_logps = get_sequence_logps(
            reference_chosen_outputs.logits,
            labels_chosen
            )

            reference_rejected_logps = get_sequence_logps(
            reference_rejected_outputs.logits,
            labels_rejected
            )

            reference_perturbed_logps = get_sequence_logps(
            reference_perturbed_outputs.logits,
            labels_perturbed
            )



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
            print(loss)

            # backward + update
            optimizer.zero_grad()
            loss.backward()
        
            
            optimizer.step()



            del policy_chosen_outputs
            del policy_rejected_outputs
            del reference_chosen_outputs
            del reference_rejected_outputs
            del policy_perturbed_outputs
            del reference_perturbed_outputs



if __name__ == "__main__":
    run()
        


