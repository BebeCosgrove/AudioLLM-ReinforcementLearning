from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
import torch

def mdpo_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor, 
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor, 
    beta = 0.1,
    reference_free: bool = False):

    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # response preference

    # image_conditional_pi_logratios = policy_chosen_logps - policy_imageless_chosen_logps
    # image_conditional_ref_logratios = reference_chosen_logps - reference_imageless_chosen_logps

    # if reference_free:
    #     image_conditional_ref_logratios = 0

    # image_conditional_logits = image_conditional_pi_logratios - image_conditional_ref_logratios  # image-conditional preference

    anchor_logits = policy_chosen_logps - reference_chosen_logps  # anchored preference

    # mDPO 
    losses = -torch.nn.functional.logsigmoid(beta * logits) \
        -torch.nn.functional.logsigmoid(beta * anchor_logits) 

    chosen_rewards = (
        beta * (policy_chosen_logps - reference_chosen_logps).detach()
        )
    rejected_rewards = (
        beta * (policy_rejected_logps - reference_rejected_logps).detach()
        )
    # imageless_rewards = (
    #     beta * (policy_imageless_chosen_logps - reference_imageless_chosen_logps).detach()
    #     )

    return losses, chosen_rewards, rejected_rewards


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

    return (token_logps * mask).sum(dim=-1) / lengths


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
        device_map= {"": "cpu"} ,
        torch_dtype=torch.float32
        
    )

    

    #chosen conversation
    chosen_conv = [[
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe what is happening in the audio."},
            {"type": "audio", "path": "/data/not_backed_up/cosgrv/af3_project/data/3-146965-A-5.wav"},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "A cat is meowing."}],
    }
    ]]

    #rejected conversation
    rejected_conv = [[
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe what is happening in the audio."},
            {"type": "audio", "path": "/data/not_backed_up/cosgrv/af3_project/data/3-146965-A-5.wav"},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "A train is running."}],
    }
    ]]

    #optimizer with 1e-7 learning rate
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-7)

    #evaluates reference and trains the policy
    reference_model.eval()
    policy_model.train()
    
    

    chosen_inputs = processor.apply_chat_template(
            chosen_conv,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            output_labels = True
        ).to(device)

    rejected_inputs = processor.apply_chat_template(
        rejected_conv,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        output_labels = True
    ).to(device)

    for step in range(10):
        dtype = next(policy_model.parameters()).dtype

        chosen_inputs["input_features"] = chosen_inputs["input_features"].to(dtype)
        rejected_inputs["input_features"] = rejected_inputs["input_features"].to(dtype)

        chosen_inputs_cpu = {
        k: v.cpu() if torch.is_tensor(v) else v
        for k, v in chosen_inputs.items()
        }

        rejected_inputs_cpu = {
        k: v.cpu() if torch.is_tensor(v) else v
        for k, v in rejected_inputs.items()
        }

        chosen_inputs_cpu["input_features"] = chosen_inputs_cpu["input_features"].float()
        rejected_inputs_cpu["input_features"] = rejected_inputs_cpu["input_features"].float()

        labels_chosen = chosen_inputs["labels"]
        labels_rejected = rejected_inputs["labels"]

        labels_chosen_cpu = labels_chosen.cpu()
        labels_rejected_cpu = labels_rejected.cpu()



        #forward pass
        policy_chosen_outputs = policy_model(**chosen_inputs)
        policy_rejected_outputs = policy_model(**rejected_inputs)

        with torch.no_grad():
            reference_chosen_outputs = reference_model(**chosen_inputs_cpu)
            reference_rejected_outputs = reference_model(**rejected_inputs_cpu)

        #logps
        policy_chosen_logps = get_sequence_logps(policy_chosen_outputs.logits, labels_chosen)
        policy_rejected_logps = get_sequence_logps(policy_rejected_outputs.logits, labels_rejected)

        reference_chosen_logps = get_sequence_logps(
        reference_chosen_outputs.logits,
        labels_chosen_cpu
        )

        reference_rejected_logps = get_sequence_logps(
        reference_rejected_outputs.logits,
        labels_rejected_cpu
        )

        reference_chosen_logps = reference_chosen_logps.to(device)
        reference_rejected_logps = reference_rejected_logps.to(device)
        # reference_chosen_logps = get_sequence_logps(reference_chosen_outputs.logits, labels_chosen)
        # reference_rejected_logps = get_sequence_logps(reference_rejected_outputs.logits, labels_rejected)

        # loss


        print("policy_chosen_logps:", policy_chosen_logps)
        print("policy_rejected_logps:", policy_rejected_logps)
        print("reference_chosen_logps:", reference_chosen_logps)
        print("reference_rejected_logps:", reference_rejected_logps)

        print("logits:", policy_chosen_logps - policy_rejected_logps
                        - (reference_chosen_logps - reference_rejected_logps))

        print("anchor:", policy_chosen_logps - reference_chosen_logps)
        losses, chosen_rewards, rejected_rewards = mdpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps
        )

        # backward + update
        optimizer.zero_grad()
        losses.backward()

        # conv1_grad = policy_model.audio_tower.conv1.weight.grad

        # print("grad min:", conv1_grad.min())
        # print("grad max:", conv1_grad.max())
        # print("grad finite:", torch.isfinite(conv1_grad).all())
    
        
        optimizer.step()

        conv1 = policy_model.audio_tower.conv1.weight

        print("weight finite:",
            torch.isfinite(conv1).all())

        print("weight min:",
            conv1.min())

        print("weight max:",
            conv1.max())

        

        # 5. logging
        print(f"step {step} loss:", losses.item())
        del policy_chosen_outputs
        del policy_rejected_outputs
        del reference_chosen_outputs
        del reference_rejected_outputs



    


    #policy outputs
    # policy_chosen_outputs = policy_model(**chosen_inputs)
    # policy_rejected_outputs = policy_model(**rejected_inputs)

    # #reference outputs
    # with torch.no_grad():
    #     reference_chosen_outputs = reference_model(**chosen_inputs)
    #     reference_rejected_outputs = reference_model(**rejected_inputs)


    # labels_chosen = chosen_inputs["labels"]
    # labels_rejected = rejected_inputs["labels"]

    # policy_chosen_logps = get_sequence_logps(policy_chosen_outputs.logits, labels_chosen)
    # policy_rejected_logps = get_sequence_logps(policy_rejected_outputs.logits, labels_rejected)
    # reference_chosen_logps = get_sequence_logps(reference_chosen_outputs.logits, labels_chosen)
    # reference_rejected_logps = get_sequence_logps(reference_rejected_outputs.logits, labels_rejected)

    # print(policy_chosen_logps - policy_rejected_logps)
    # print(reference_chosen_logps - reference_rejected_logps)



    # losses, chosen_rewards, rejected_rewards = mdpo_loss(
    #     policy_chosen_logps,
    #     policy_rejected_logps,
    #     reference_chosen_logps,
    #     reference_rejected_logps
    # )

    # print(chosen_rewards)
    # print(rejected_rewards)




if __name__ == "__main__":
    run()
        


