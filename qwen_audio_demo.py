import torch
import soundfile as sf
from transformers import AutoProcessor, AutoModelForCausalLM

model_id = "Qwen/Qwen2-Audio-7B-Instruct"

print("Loading model...")

processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

audio, sr = sf.read("data/example.wav")

prompt = "Describe the audio in detail."

inputs = processor(
    text=prompt,
    audios=audio,
    sampling_rate=sr,
    return_tensors="pt"
)

inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

print("Running inference...")

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=100)

print(processor.batch_decode(output, skip_special_tokens=True)[0])
