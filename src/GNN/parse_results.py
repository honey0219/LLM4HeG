import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    # training arguments
    parser.add_argument('--result_path', type=str, required=True,
                        help="Path to results txt.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    best_mean_score = 0
    best_params = {}
    cur_lr, cur_wd, cur_scores = None, None, []
    best_lr, best_wd, best_scores = None, None, []
  
    with open(args.result_path, 'r') as f:
        for line in f:
            if line.startswith('LR'):
                cur_lr = float(line.split('=')[-1])
            elif line.startswith('WD'):
                cur_wd = float(line.split('=')[-1])
            elif line.startswith('Split'):
                split_id = float(line.split('=')[-1])
            elif line.startswith('0.'):
                res = float(line.split(' ')[-1])
                cur_scores.append(res)
                if split_id == 9:
                    if np.mean(cur_scores) > best_mean_score:
                        best_mean_score = np.mean(cur_scores)
                        best_scores = cur_scores
                        best_params = {'lr': cur_lr, 'wd': cur_wd}
                    cur_scores = []
    
    print(f'{100 * np.mean(best_scores):.2f}±{100 * np.std(best_scores):.2f}')
    print(best_params)

            