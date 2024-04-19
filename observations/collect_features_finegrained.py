import json
from fastchat.model import load_model, get_conversation_template
from models.modeling_mistral_benchmark_finegrained import MistralForCausalLM as MyMistralForCausalLM
import torch
import transformers
from tqdm import tqdm
import numpy as np
import argparse

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                pass
    return data

def load_data(data, model_name, tokenizer):
    conv = get_conversation_template(model_name)
    max_turns = len(data) // 2
    for i in range(max_turns - 1):
        conv.append_message(conv.roles[0], data[i*2])
        conv.append_message(conv.roles[1], data[i*2+1])
    conv.append_message(conv.roles[0], data[(max_turns - 1)*2])
    start = tokenizer.encode(conv.get_prompt(), return_tensors='pt').shape[1]
    conv.append_message(conv.roles[1], data[(max_turns - 1)*2+1])
    end = tokenizer.encode(conv.get_prompt(), return_tensors='pt').shape[1]
    return conv.get_prompt(), start - 1, end

def main(args):
    dataset_path = args.dataset_path
    dataset = load_jsonl(dataset_path)
    model_name = args.model_name
    model = MyMistralForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        # use_flash_attention_2=True
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
        use_fast=False,
    )

    features = []
    window_size = args.window_size
    steps = args.steps
    threshold = args.threshold
    top_k = args.top_k
    layer_len = len(model.model.layers)

    for data in tqdm(dataset):
        features_per_data = []
        with torch.inference_mode():
            prompt, prev_len, end = load_data(data, model_name, tokenizer)
            input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
            for layer_id in range(layer_len):
                model.model.layers[layer_id].self_attn.prev_len = prev_len
                model.model.layers[layer_id].self_attn.steps = steps
                model.model.layers[layer_id].self_attn.threshold = threshold
                model.model.layers[layer_id].self_attn.window_size = window_size
                model.model.layers[layer_id].self_attn.top_k = top_k
            outputs = model(input_ids, output_attentions = False, use_cache = False)
            for layer_id in range(layer_len):
                features_per_data.append(model.model.layers[layer_id].self_attn.features_per_data)
                # attn_weights = model.model.layers[layer_id].self_attn.attn_weights
                # total_len = attn_weights.shape[-1]
                # for step in range(steps):
                #     start = prev_len - window_size
                #     end = prev_len
                #     shift = window_size * step
                #     prev_attn_sum = attn_weights[0, :, start:end, :start].sum(1)
                #     cur_attn_sum = attn_weights[0, :, start + shift + window_size:end + shift + window_size, :start].sum(1)
                #     prev_attn_sum_threshold = prev_attn_sum > (threshold * window_size)
                #     cur_attn_sum_threshold = cur_attn_sum > (threshold * window_size)
                #     activation_overlap = (prev_attn_sum_threshold & cur_attn_sum_threshold).sum(-1)
                #     activation_sum = cur_attn_sum_threshold.sum(-1)
                #     hit_rate = activation_overlap / activation_sum
                #     hit_rate = hit_rate.mean()
                    
                #     features_per_data.append(hit_rate.item())
                # total_len = attn_weights.shape[-1]
                # for step in range(steps):
                #     activation_overlaps = []
                #     for channel_id in range(attn_weights.shape[1]):
                #         start = prev_len - window_size
                #         end = prev_len
                #         shift = window_size * step
                #         prev_attn_sum = attn_weights[0, channel_id, start:end, :start].sum(0)
                #         cur_attn_sum = attn_weights[0, channel_id, start + shift + window_size:end + shift + window_size, :start].sum(0)
                #         activation_overlap = ((prev_attn_sum > threshold) & (cur_attn_sum > threshold)).sum()/(cur_attn_sum > threshold).sum()
                #         # check if nan skip
                #         if not torch.isnan(activation_overlap):
                #             activation_overlaps.append(activation_overlap.item())

                #     features_per_data.append(np.mean(activation_overlaps))
        # jsonl line by line
        with open(args.output_path, 'a') as f:
            f.write(json.dumps(features_per_data) + '\n')
    # np.save(args.output_path, np.array(features))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default = 'mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument("--window_size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--top_k", type=int, default=0.05)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args)