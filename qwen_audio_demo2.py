import torch
import soundfile as sf
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
import librosa

model_id = "Qwen/Qwen2-Audio-7B-Instruct"

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

model = Qwen2AudioForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
#import pdb; pdb.set_trace()

audio, sr = sf.read("/data/not_backed_up/cosgrv/af3_project/data/kwcjvoqsu_M_30.wav")

if len(audio.shape) > 1:
    audio = audio.mean(axis=1)

audio = audio.astype("float32")

audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
sr = 16000


conversation = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio, "sampling_rate": sr},
            {"type": "text", "text": "Describe the audio."}
        ]
    }
]

inputs = processor(
    text=processor.apply_chat_template(conversation, tokenize=False),
    audio=audio,
    sampling_rate=sr,
    return_tensors="pt"
)

inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=100)

print(processor.batch_decode(output, skip_special_tokens=True)[0])