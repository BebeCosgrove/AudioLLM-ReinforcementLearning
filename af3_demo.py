import torch
import soundfile as sf
from transformers import AutoProcessor, AutoModel
import librosa

device = "cuda"


model_id = "nvidia/audio-flamingo-3-hf"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
).to(device)

model.eval()

# 2. Load audio
audio, sr = sf.read("/data/not_backed_up/cosgrv/af3_project/data/kwcjvoqsu_M_30.wav")

if len(audio.shape) > 1:
    audio = audio.mean(axis=1)

audio = audio.astype("float32")

audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
sr = 16000


prompt = "<sound>\nDescribe the audio in exactly one short sentence."

inputs = processor(
    text=prompt,
    audio=audio,
    sampling_rate=sr,
    return_tensors="pt"
).to(device)




inputs = {k: v.to(model.device) for k, v in inputs.items()}




output = model.generate(
    **inputs,
    max_new_tokens=30,
    do_sample=False,
    eos_token_id=processor.tokenizer.eos_token_id,
)

generated = output[:, inputs["input_ids"].shape[1]:]
print(processor.batch_decode(generated, skip_special_tokens=True)[0])




