from accelerate import Accelerator
import torch

accelerator = Accelerator()

print(
    f"rank={accelerator.process_index}, "
    f"local_rank={accelerator.local_process_index}, "
    f"device={accelerator.device}, "
    f"world_size={accelerator.num_processes}",
    flush=True,
)

x = torch.tensor([accelerator.process_index], device=accelerator.device)
accelerator.wait_for_everyone()
