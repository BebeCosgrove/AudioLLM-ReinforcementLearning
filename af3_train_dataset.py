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
        prompt_convs = []
        perturbed_prompt_convs = []

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

            prompt_convs.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe what is happening in the audio."},
                        {"type": "audio", "path": ex["audio"]}
                    ]
                }
            ])

            perturbed_prompt_convs.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe what is happening in the audio."},
                        {
                            "type": "audio",
                            "path": ex["perturbed_audio"]
                        }
                    ]
                }
            ])

        #import pdb; pdb.set_trace()
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

        #gets the index of the start of the response
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
        

        #makes everything after response start their actual input_ids
        for b in range(len(examples)):
            prompt_mask = prompt_inputs["attention_mask"][b]

            prompt_start = prompt_mask.nonzero()[0].item() # gets the index of where the padding stops

            response_start = prompt_start + prompt_lengths[b]

            chosen_labels[b, response_start:] = (
                chosen_inputs["input_ids"][b, response_start:]
            )
        for b in range(len(examples)):
            prompt_mask = prompt_inputs["attention_mask"][b]

            prompt_start = prompt_mask.nonzero()[0].item()

            response_start = prompt_start + prompt_lengths[b]

            rejected_labels[b, response_start:] = (
                rejected_inputs["input_ids"][b, response_start:]
            )
        for b in range(len(examples)):
            prompt_mask = perturbed_prompt_inputs["attention_mask"][b]

            prompt_start = prompt_mask.nonzero()[0].item()

            response_start = prompt_start + perturbed_prompt_lengths[b]

            perturbed_labels[b, response_start:] = (
                perturbed_inputs["input_ids"][b, response_start:]
            )

        import pdb; pdb.set_trace()

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


def find_assistant_start(input_ids, tokenizer):
    assistant_token_id = tokenizer.encode(
        "assistant",
        add_special_tokens=False
    )[0]

    # Search from the end backwards — find the LAST occurrence
    for i in range(len(input_ids) - 1, -1, -1):
        if input_ids[i].item() == assistant_token_id:
            return i + 2  # skip "assistant" and newline


# #goes from end of labels and finds first index of the response
# def find_response_start(labels, example):
#     non_neg = (labels != -100).nonzero(as_tuple=True)[example]
#     # find where the last continuous block starts
#     for i in range(len(non_neg) - 1, -1, -1):
#         if i == 0 or non_neg[i] - non_neg[i-1] > 1:
#             return non_neg[i].item()


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
    tokenizer = processor.tokenizer

    #evaluates reference and trains the policy
    reference_model.eval()
    policy_model.train()
    

    for epoch in range(10):
        for batch in loader:

            #gets the input sequence from that batch
            chosen_inputs = batch["chosen"]
            rejected_inputs = batch["rejected"]
            perturbed_inputs = batch["perturbed"] 
            prompt_only = batch["prompt"] 

            # print(tokenizer.decode(chosen_inputs["input_ids"][0][268:275]))
            # print(tokenizer.decode(chosen_inputs["input_ids"][1][268:275]))

            #import pdb; pdb.set_trace()

            chosen_inputs["input_features"] = (
            chosen_inputs["input_features"].to(torch.bfloat16)
            )

            rejected_inputs["input_features"] = (
                rejected_inputs["input_features"].to(torch.bfloat16)
            )

            perturbed_inputs["input_features"] = (
                perturbed_inputs["input_features"].to(torch.bfloat16)
            )

            #clones the labels and assigns all of them to -100
            response_chosen_ids = chosen_inputs["input_ids"].clone()
            response_chosen_ids[:] = -100

            response_rejected_ids = rejected_inputs["input_ids"].clone()
            response_rejected_ids[:] = -100

            response_perturbed_ids = perturbed_inputs["input_ids"].clone()
            response_perturbed_ids[:] = -100


            #changes the cloned id holder so that only non -100 places are the response
            for b in range(chosen_inputs["input_ids"].size(0)):

                chosen_input_ids = chosen_inputs["input_ids"][b]
                rejected_input_ids = rejected_inputs["input_ids"][b]
                perturbed_input_ids = perturbed_inputs["input_ids"][b]

                chosen_assistant_place = find_assistant_start(chosen_input_ids, tokenizer)
                rejected_assistant_place = find_assistant_start(rejected_input_ids, tokenizer)
                perturbed_assistant_place = find_assistant_start(perturbed_input_ids, tokenizer)

                chosen_tokens = chosen_input_ids[chosen_assistant_place:]
                response_chosen_ids[b, chosen_assistant_place:chosen_assistant_place + len(chosen_tokens)] = chosen_tokens

        
                rejected_tokens = rejected_input_ids[rejected_assistant_place:]
                response_rejected_ids[b, rejected_assistant_place:rejected_assistant_place + len(rejected_tokens)] = rejected_tokens


                perturbed_tokens = perturbed_input_ids[perturbed_assistant_place:]
                response_perturbed_ids[b, perturbed_assistant_place:perturbed_assistant_place + len(perturbed_tokens)] = perturbed_tokens
                
            #shows that ids only contains the response and everything else is -100
            # for b in range(chosen_inputs["input_ids"].size(0)):
            #     valid_tokens = response_chosen_ids[b]
            #     valid_tokens = valid_tokens[valid_tokens != -100]
            #     print(f"Chosen response {b}:", tokenizer.decode(valid_tokens))

            #     valid_tokens = response_rejected_ids[b]
            #     valid_tokens = valid_tokens[valid_tokens != -100]
            #     print(f"Rejected response {b}:", tokenizer.decode(valid_tokens))

            #     valid_tokens = response_perturbed_ids[b]
            #     valid_tokens = valid_tokens[valid_tokens != -100]
            #     print(f"Perturbed response {b}:", tokenizer.decode(valid_tokens))

            # double check by pdb together.
                
            print(policy_model.device)
                
            #pops labels and moves the tensors to cpu
            chosen_inputs.pop("labels", None)
            rejected_inputs.pop("labels", None)
            perturbed_inputs.pop("labels", None)

            chosen_inputs = {k: v.to(policy_model.device) if isinstance(v, torch.Tensor) else v 
                             for k, v in chosen_inputs.items()}
            rejected_inputs = {k: v.to(policy_model.device) if isinstance(v, torch.Tensor) else v 
                               for k, v in rejected_inputs.items()}
            perturbed_inputs = {k: v.to(policy_model.device) if isinstance(v, torch.Tensor) else v 
                                for k, v in perturbed_inputs.items()}

            response_chosen_ids = response_chosen_ids.to(policy_model.device)
            response_rejected_ids = response_rejected_ids.to(policy_model.device)
            response_perturbed_ids = response_perturbed_ids.to(policy_model.device)

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
            policy_chosen_logps = get_sequence_logps(policy_chosen_outputs.logits, response_chosen_ids)
            policy_rejected_logps = get_sequence_logps(policy_rejected_outputs.logits, response_rejected_ids)
            policy_perturbed_logps = get_sequence_logps(policy_perturbed_outputs.logits, response_perturbed_ids)

            reference_chosen_logps = get_sequence_logps(
            reference_chosen_outputs.logits,
            response_chosen_ids
            )

            reference_rejected_logps = get_sequence_logps(
            reference_rejected_outputs.logits,
            response_rejected_ids
            )

            reference_perturbed_logps = get_sequence_logps(
            reference_perturbed_outputs.logits,
            response_perturbed_ids
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
            #print(loss)

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
        


