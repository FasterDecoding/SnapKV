import json
import argparse
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

def collect_prompt(dataset_id, data_info, prev_range = [2000, 2500], min_length = 64 * 8):
    prompt_list = []
    for info in data_info:
        for i, turn in enumerate(info['turns']):
            start, end = turn
            if start > prev_range[0] and start < prev_range[1] and end - start > min_length:
                prompt_list.append({'idx': info['idx'], 'turn': i, 'start': start, 'end': end, 'dataset_id':dataset_id})
    return prompt_list

# main
def main(args):
    data_info_path = args.data_info_path
    dataset_id_begin = args.dataset_id_begin
    dataset_id_end = args.dataset_id_end
    min_length = args.min_length
    length_start = args.length_start
    length_end = args.length_end
    length_step = args.length_step
    total_step = (length_end - length_start) // length_step + 1
    for dataset_id in range(dataset_id_begin, dataset_id_end + 1):
        data_info = load_jsonl(data_info_path.replace('DATASET_ID', str(dataset_id)))
        for step in range(total_step):
            cur_start = length_start + step * length_step
            cur_end = cur_start + length_step if step < total_step - 1 else 100000
            prompt_list = collect_prompt(dataset_id, data_info, [cur_start, cur_end], min_length)
            with open(f'./data/filtered_info/prompt_{dataset_id}_len_{cur_start}_{cur_end}.json', 'w') as f:
                json.dump(prompt_list, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_info_path', type=str, default='./data/ultrachat_DATASET_ID_categorized.jsonl')
    parser.add_argument('--dataset_id_begin', type=int, default=0)
    parser.add_argument('--dataset_id_end', type=int, default=9)
    parser.add_argument('--min_length', type=int, default=64 * 8)
    parser.add_argument('--length_start', type=int, default=1000)
    parser.add_argument('--length_end', type=int, default=3000)
    parser.add_argument('--length_step', type=int, default=500)
    args = parser.parse_args()
    main(args)