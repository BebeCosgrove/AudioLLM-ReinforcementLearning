import json

with open("/data/not_backed_up/cosgrv/af3_project/ah_existence/perturbed_datasets/no_audio_ah_existence.json") as f:
    data = json.load(f)

for fold in range(1, 6):

    with open(f"balanced_splits_data/balanced_split_{fold}.json") as f:
        split = json.load(f)

    train_data = []
    for i in split["train"]:
        train_data.append(data[i])

    val_data = []
    for i in split["val"]:
        val_data.append(data[i])

    test_data = []
    for i in split["test"]:
        test_data.append(data[i])



    OUTPUT_DIR = "/data/not_backed_up/cosgrv/af3_project/ah_existence/perturbed_split_data/"
    with open(f"{OUTPUT_DIR}no_audio_train_fold{fold}.json", "w") as f:
        json.dump(train_data, f, indent=2)

    with open(f"{OUTPUT_DIR}no_audio_val_fold{fold}.json", "w") as f:
        json.dump(val_data, f, indent=2)

    with open(f"{OUTPUT_DIR}no_audio_test_fold{fold}.json", "w") as f:
        json.dump(test_data, f, indent=2)



