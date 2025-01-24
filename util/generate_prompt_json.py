import numpy as np
import json
import os
import argparse

prompt_info = {
    'Cornell': {
        'BT': "Background: I have a dataset containing web page information collected from computer science department websites of various universities. These web pages have been manually categorized into five categories, including student, staff, faculty, course, and project." + "Task: I will provide you with the content of two web pages, and I want you to determine if they belong to the same category among student, staff, course, faculty, and project.",
        'A': ". Answer template: \"Yes\"  or \"No\" Please think step by step.",
        'F': "The first web page: ",
        'S': "The second web page: "
    },
}

def nhoodSplit(adj: np.ndarray, nhood):
    assert adj.ndim == 2 and adj.shape[0] == adj.shape[1]
    if np.isnan(nhood):
        return np.ones(adj.shape)
    mt = np.eye(adj.shape[1])
    mtList = [mt]
    i = 0
    edge_sum = 0
    while i < nhood:
        prev_mt = mt
        mt = mt @ (adj + np.eye(adj.shape[0]))
        mt = (mt > 0).astype(mt.dtype)
        new_edge_sum = np.sum(mt)
        if edge_sum == new_edge_sum:
            break
        else:
            edge_sum = new_edge_sum
        i += 1
        mtList.append(mt - prev_mt)
    return mtList

def truncate_tokens(text, max_tokens=1000):
    tokens = text.split()
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return ' '.join(tokens)

def generate_prompt_json(data_name, mode, save_dir):
    npz_data = np.load(f"../dataset/{data_name}.npz")
    edge = npz_data['edges']
    labels = npz_data['node_labels']
    features = npz_data['node_features']

    train_mask = npz_data['train_masks']
    val_mask = npz_data['val_masks']
    test_mask = npz_data['test_masks']
    text_list = npz_data['node_text']

    mode_path = {
        "train_all": "train.json",
        "train_two_hop": "train.json",
        "train_one_hop": "train.json",
        "one_hop_infer": "one_hop_infer.json",
        "two_hop_infer": "two_hop_infer.json",
        "test_one_hop": "test.json",
        "test_two_hop": "test.json",
        "val_two_hop": "val.json",
        "val_one_hop": "val.json",
    }


    num_nodes = len(labels)
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(len(edge[0])):
        start_node = edge[0, i]
        end_node = edge[1, i]
        adj_matrix[start_node, end_node] = 1
        adj_matrix[end_node, start_node] = 1
    list_result = nhoodSplit(adj_matrix, nhood=2)
    nonzero_counts = [np.count_nonzero(matrix) for matrix in list_result]
    print("Number of non-zero elements in the one-hop and two-hop adjacency matrix:", nonzero_counts)


    if mode == "one_hop_infer":
        combined = list_result[1]
    elif mode == "two_hop_infer":
        combined = np.maximum(list_result[1], list_result[2])
    elif mode == "train_all":
        combined = np.ones((num_nodes, num_nodes))
        train_nodes = np.where(train_mask)[0]
        masked_combined = np.zeros_like(combined)
        masked_combined[np.ix_(train_nodes, train_nodes)] = combined[np.ix_(train_nodes, train_nodes)]
        combined = masked_combined
    elif mode == "train_one_hop":
        combined = list_result[1]
        train_nodes = np.where(train_mask)[0]
        masked_combined = np.zeros_like(combined)
        masked_combined[np.ix_(train_nodes, train_nodes)] = combined[np.ix_(train_nodes, train_nodes)]
        combined = masked_combined
    elif mode == "train_two_hop":
        combined = np.maximum(list_result[1], list_result[2])
        train_nodes = np.where(train_mask)[0]
        masked_combined = np.zeros_like(combined)
        masked_combined[np.ix_(train_nodes, train_nodes)] = combined[np.ix_(train_nodes, train_nodes)]
        combined = masked_combined
    elif mode == "test_one_hop":
        combined = list_result[1]
        test_nodes = np.where(test_mask)[0]
        masked_combined = np.zeros_like(combined)
        masked_combined[np.ix_(test_nodes, test_nodes)] = combined[np.ix_(test_nodes, test_nodes)]
        combined = masked_combined
    elif mode == "test_two_hop":
        combined = np.maximum(list_result[1], list_result[2])
        test_nodes = np.where(test_mask)[0]
        masked_combined = np.zeros_like(combined)
        masked_combined[np.ix_(test_nodes, test_nodes)] = combined[np.ix_(test_nodes, test_nodes)]
        combined = masked_combined
    elif mode == "val_one_hop":
        combined = list_result[1]
        val_nodes = np.where(val_mask)[0]
        masked_combined = np.zeros_like(combined)
        masked_combined[np.ix_(val_nodes, val_nodes)] = combined[np.ix_(val_nodes, val_nodes)]
        combined = masked_combined
    elif mode == "val_two_hop":
        combined = np.maximum(list_result[1], list_result[2])
        val_nodes = np.where(val_mask)[0]
        masked_combined = np.zeros_like(combined)
        masked_combined[np.ix_(val_nodes, val_nodes)] = combined[np.ix_(val_nodes, val_nodes)]
        combined = masked_combined
    else:
        raise ValueError(f"The mode '{mode}' is not valid.")
    nonzero_count = np.count_nonzero(combined)
    print("Number of non-zero elements in the target template matrix:", nonzero_count)
    is_symmetric = np.array_equal(combined, combined.T)
    print("Is the target template matrix symmetric:", is_symmetric)

    prompt_list = []
    id = 0
    for start_node in range(num_nodes):
        for end_node in range(start_node + 1, num_nodes):
            if combined[start_node, end_node] <= 0:
                continue
            if labels[start_node] == labels[end_node]:
                gpt = "Yes"
            else:
                gpt = "No"
            temp = \
                {
                    "id": str(id),
                    "conversations": [
                        {"from": "human",
                         "value": prompt_info[data_name]['BT'] + prompt_info[data_name]['F'] + truncate_tokens(text_list[start_node]) + prompt_info[data_name]['S'] + truncate_tokens(text_list[end_node])+ prompt_info[data_name]['A']},
                        {"from": "gpt", "value": gpt}
                    ]
                }
            prompt_list.append(temp)
            id += 1
    print("template length：", len(prompt_list))
    print("first template: ", prompt_list[0])
    # dir = '../llm_pred/prompt_json/' + data_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("save to : " + save_dir + '/' + mode_path[mode])
    with open(save_dir + '/' + mode_path[mode], "w") as fout:
        json.dump(prompt_list, fout, indent=2, ensure_ascii=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    args = parser.parse_args()

    generate_prompt_json(args.data_name, args.mode, args.save_dir)