# Exploring the Potential of Large Language Models for Heterophilic Graphs

## Data

Each dataset in the `dataset` directory is in `.npz` format and includes:
`edges`, `node_labels`, `node_features`, `node_texts`, `label_texts`, `train_masks`, `val_masks`, `test_masks`.

## Experiments

### Stage1

Set up the environment following the `README.md` files in the `src/LLM` and `src/SLM` directories.

#### Finetune Vicuna 7B

Generate training and inference data using`util/generate_prompt_json.py`

Fine-tune the Vicuna 7B model, merge it with any additional components, and then obtain the inference results:
```bash
cd src/LLM
bash scripts/train_lora.sh
python fastchat/model/apply_lora.py
bash eval.sh
```
Convert the inference results to the Stage 2 data format with`util/llm_result.py`

#### Distill SLMs

Generate the required validation and test data for distillation using the `util/generate_prompt_json.py` and obtain inference results from the fine-tuned Vicuna 7B model.

Create summary distillation training data with using `util/get_distill_json.py`. Register this data in `src/SLM/data/dataset_info.json`.

Fine-tune the SLM model, merge it with any additional components, and then obtain the inference results:
```bash
cd src/SLM
llamafactory-cli train examples/train_lora/train.yaml
llamafactory-cli export examples/merge_lora/merge.yaml
llamafactory-cli train examples/inference/inference.yaml
```
Finally, run `util/slm_result.py` to obtain the data format required for Stage 2.

### Stage2

#### Installation

Install the required packages using the dependencies listed in `requirements.txt`.

#### Run

To run experiments, execute the `run_train.sh` script located in the `src/GNN` directory.

To compute averaged statistics, use the `src/parse_results.py` script with the `--result_path` argument, providing the path to your experiment results.



