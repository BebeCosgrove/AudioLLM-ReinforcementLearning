import json
import random
from collections import defaultdict

INPUT_JSON = "/data/not_backed_up/cosgrv/af3_project/ah_existence/perturbed_datasets/no_audio_ah_existence.json"

TRAIN_JSON = "/data/not_backed_up/cosgrv/af3_project/ah_existence/perturbed_datasets/ah_existence_no_audio_train.json"
VAL_JSON = "/data/not_backed_up/cosgrv/af3_project/ah_existence/perturbed_datasets/ah_existence_no_audio_val.json"
TEST_JSON = "/data/not_backed_up/cosgrv/af3_project/ah_existence/perturbed_datasets/ah_existence_no_audio_test.json"

# Load dataset
with open(INPUT_JSON) as f:
    data = json.load(f)

# Group by audio file
groups = defaultdict(list)

for item in data:
    groups[item["path"]].append(item)

paths = list(groups.keys())

random.seed(42)
random.shuffle(paths)

n_paths = len(paths)

train_end = int(0.8 * n_paths)
val_end = int(0.9 * n_paths)

train_paths = set(paths[:train_end])
val_paths = set(paths[train_end:val_end])
test_paths = set(paths[val_end:])

train_data = []
val_data = []
test_data = []

for path, examples in groups.items():
    if path in train_paths:
        train_data.extend(examples)
    elif path in val_paths:
        val_data.extend(examples)
    else:
        test_data.extend(examples)

with open(TRAIN_JSON, "w") as f:
    json.dump(train_data, f, indent=2)

with open(VAL_JSON, "w") as f:
    json.dump(val_data, f, indent=2)

with open(TEST_JSON, "w") as f:
    json.dump(test_data, f, indent=2)

print(f"Train: {len(train_data)} examples")
print(f"Val:   {len(val_data)} examples")
print(f"Test:  {len(test_data)} examples")

print(f"Unique train audio: {len(train_paths)}")
print(f"Unique val audio:   {len(val_paths)}")
print(f"Unique test audio:  {len(test_paths)}")