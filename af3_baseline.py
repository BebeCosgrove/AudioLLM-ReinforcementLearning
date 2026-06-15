from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
import librosa
import os


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
        prompt_convs = []
        chosen_response_convs = []

        BASE_DIR = "/data/not_backed_up/cosgrv/af3_project/ah_existence"

        for ex in examples:
            audio_path = os.path.join(BASE_DIR, ex["path"])

            audio = librosa.load(audio_path, sr=16000, mono=True)[0]

            chosen_convs.append([
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

            prompt_convs.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ex["Q"]},
                        {"type": "audio", "path": audio}
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


        #import pdb; pdb.set_trace()
        chosen_inputs = self.processor.apply_chat_template(
        chosen_convs,
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


        #gets the length of the prompt
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1)
        #perturbed_prompt_lengths = (perturbed_prompt_inputs["attention_mask"].sum(dim=1)
                                    

        
        #makes a copy of chosen_inputs where everything is -100
        chosen_labels = torch.full_like(
        chosen_inputs["input_ids"],
        -100
    )
   

        #makes everything after response have their actual input_ids instead of -100
        for b in range(len(examples)):
            chosen_mask = chosen_inputs["attention_mask"][b]
            chosen_start = chosen_mask.nonzero()[0].item()
            prompt_len = prompt_lengths[b].item()
            response_start = chosen_start + prompt_len
            chosen_labels[b, response_start:] = chosen_inputs["input_ids"][b, response_start:]
        

        

        #assigns the labels to the new fixed labels that only have ids for the response
        chosen_inputs["labels"] = chosen_labels
        


        return {
        "chosen": chosen_inputs
        }


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

    with open("/data/not_backed_up/cosgrv/af3_project/ah_existence/perturbed_datasets/ah_existence_no_audio_train.json") as f:
        data = json.load(f)


    #gets train dataset to pytorch form
    train_dataset = AudioDPODataset(data)

    #gets collator with our type of processor
    collator= AudioMDPOCollator(processor)

    # get dataloader
    loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn= collator
    )

    
    #optimizer with 1e-6 learning rate
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-6)
    tokenizer = processor.tokenizer

    #evaluates reference and trains the policy
    reference_model.eval()
    policy_model.train()
    

    for epoch in range(50):
        for batch in loader:

            #gets the input sequence from that batch
            chosen_inputs = batch["chosen"]


            chosen_inputs["input_features"] = (
            chosen_inputs["input_features"].to(torch.bfloat16)
            )

                
            #pops labels and moves the tensors to gpu
            chosen_labels = chosen_inputs.pop("labels")

            #moves to gpu
            chosen_inputs = {k: v.to(policy_model.device) if isinstance(v, torch.Tensor) else v 
                             for k, v in chosen_inputs.items()}
            
            #moves to gpu
            chosen_labels = chosen_labels.to(policy_model.device)

            #forward pass
            policy_chosen_outputs = policy_model(**chosen_inputs)

            #freezes reference model
            with torch.no_grad():
                reference_chosen_outputs = reference_model(**chosen_inputs)

            #logps
            policy_chosen_logps = get_sequence_logps(policy_chosen_outputs.logits, chosen_labels)

            reference_chosen_logps = get_sequence_logps(reference_chosen_outputs.logits, chosen_labels)



    

            #gets average of the batches losses
            loss = losses.mean()
            chosen_reward_mean = chosen_rewards.mean().item()

            print(f"Epoch {epoch} | Loss: {loss.item():.4f} | "
                f"Chosen reward: {chosen_reward_mean:.4f} | "
                )

            # backward + update
            optimizer.zero_grad()
            loss.backward()
        
            
            optimizer.step()




if __name__ == "__main__":
    run()
        


