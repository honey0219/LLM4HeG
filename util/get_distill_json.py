import json
import argparse

def get_distill_json(val_truth_path, val_result_path, test_truth_path, test_result_path, train_truth_path, save_path):
    merged_data = []
    with open(val_result_path, "r") as file:
        result_val = json.load(file)
    with open(val_truth_path, "r") as file:
        truth_val = json.load(file)

    for i in range(len(truth_val)):
        truth_val[i]['conversations'][1]['value'] = result_val[i]['res']

    merged_data.extend(truth_val)

    with open(test_result_path, "r") as file:
        result_test = json.load(file)
    with open(test_truth_path, "r") as file:
        truth_test = json.load(file)

    for i in range(len(truth_test)):
        truth_test[i]['conversations'][1]['value'] = result_test[i]['res']

    merged_data.extend(truth_test)


    with open(train_truth_path, 'r') as f:
        train_truth = json.load(f)

    merged_data.extend(train_truth)
    print("length: ", len(merged_data))

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_truth_path", type=str, required=True)
    parser.add_argument("--val_result_path", type=str, required=True)
    parser.add_argument("--test_truth_path", type=str, required=True)
    parser.add_argument("--test_result_path", type=str, required=True)
    parser.add_argument("--train_truth_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)

    args = parser.parse_args()

    get_distill_json(args.val_truth_path, args.val_result_path, args.test_truth_path, args.test_result_path, args.train_truth_path, args.save_path)
