import json
import random

INPUT_JSON = "/data/not_backed_up/cosgrv/af3_project/ah_existence/perturbed_datasets/no_audio_ah_existence.json"

TRAIN_JSON = "/data/not_backed_up/cosgrv/af3_project/ah_existence/perturbed_datasets/ah_existence_no_audio_train.json"
VAL_JSON = "/data/not_backed_up/cosgrv/af3_project/ah_existence/perturbed_datasets/ah_existence_no_audio_val.json"
TEST_JSON = "/data/not_backed_up/cosgrv/af3_project/ah_existence/perturbed_datasets/ah_existence_no_audio_test.json"

# Load dataset
with open(INPUT_JSON) as f:
    data = json.load(f)

# Shuffle reproducibly
random.seed(42)
random.shuffle(data)

n = len(data)

train_end = int(0.8 * n)
val_end = int(0.9 * n)

train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]

# Save splits
with open(TRAIN_JSON, "w") as f:
    json.dump(train_data, f, indent=2)

with open(VAL_JSON, "w") as f:
    json.dump(val_data, f, indent=2)

with open(TEST_JSON, "w") as f:
    json.dump(test_data, f, indent=2)

print(f"Train: {len(train_data)}")
print(f"Val:   {len(val_data)}")
print(f"Test:  {len(test_data)}")