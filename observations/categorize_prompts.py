import transformers
import torch
import json
from fastchat.model import load_model, get_conversation_template
import argparse
from tqdm import tqdm
# load jsonl dataset
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                pass
    return data

def classify_per_data(dataset, idx, tokenizer, model_name):
    data = dataset[idx]['data']
    max_turns = len(data) // 2
    conv = get_conversation_template(model_name)
    info = {}
    info['max_turns'] = max_turns
    info['idx'] = idx
    info['id'] = dataset[idx]['id']
    info['turns'] = []
    for i in range(max_turns):
        conv.append_message(conv.roles[0], data[i*2])
        # get start stamp
        start = tokenizer.encode(conv.get_prompt(), return_tensors='pt').shape[1]
        conv.append_message(conv.roles[1], data[i*2+1])
        # get end stamp
        end = tokenizer.encode(conv.get_prompt(), return_tensors='pt').shape[1]
        info['turns'].append((start, end))
    return info

def main(args):
    model_name = args.model_name
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    dataset = load_jsonl(args.dataset_path)
    for idx in tqdm(range(len(dataset))):
        info = classify_per_data(dataset, idx, tokenizer, model_name)
        # save info to jsonl by line
        with open(args.output_path, 'a') as f:
            f.write(json.dumps(info) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Categorize prompts in a dataset")
    # add arguments
    parser.add_argument('--dataset_path', type=str, help='path to the dataset')
    parser.add_argument('--output_path', type=str, help='path to the output')
    parser.add_argument('--model_name', type=str, help='model name')
    # get args
    args = parser.parse_args()
    main(args)