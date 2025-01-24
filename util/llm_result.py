import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os
import argparse


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


def llm_result(data_name, mode, truth_path, result_path, save_dir):
    npz_data = np.load(f"../dataset/{data_name}.npz")
    edge = npz_data['edges']
    labels = npz_data['node_labels']


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

    with open(truth_path, 'r', encoding='utf-8') as f:
        data_true = json.load(f)

    truth_answers = {}
    for item in data_true:
        id = item['id']
        gpt_answer = item['conversations'][-1]['value']  # 获取最后一次对话中的回答
        truth_answers[id] = gpt_answer

    with open(result_path, 'r', encoding='utf-8') as f:
        data_res = json.load(f)

    result_dict = {}
    for item in data_res:
        id_int = item['id']
        res_value = item['res']
        result_dict[id_int] = res_value

    result = [1 if "Yes" in res else 0 for id, res in result_dict.items()]
    truth = [1 if "Yes" in res else 0 for id, res in truth_answers.items()]
    accuracy = accuracy_score(truth, result)
    precision = precision_score(truth, result)
    recall = recall_score(truth, result)
    f1 = f1_score(truth, result)

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    print("result length", len(result))
    id = 0
    for start_node in range(num_nodes):
        for end_node in range(start_node + 1, num_nodes):
            if combined[start_node, end_node] <= 0:
                combined[start_node, end_node] = 0
                continue
            combined[start_node, end_node] = result[id]
            combined[end_node, start_node] = result[id]
            id += 1
    print(id)
    mask = np.eye(combined.shape[0], dtype=bool)
    combined[mask] = 1
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(save_dir+'/result.npy', combined)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--truth_path", type=str, required=True)
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    args = parser.parse_args()

    llm_result(args.data_name, args.mode, args.truth_path, args.result_path, args.save_dir)

