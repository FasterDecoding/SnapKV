import json
import random

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                pass
    return data
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
def collect_random_prompts(filter_path, dataset_id_begin, dataset_id_end, start, end, total_num = 5000):
    summary_random_info = {}
    prompts = []
    data_info = []
    for dataset_id in range(dataset_id_begin, dataset_id_end + 1):
        cur_data_info = load_json(filter_path.replace('DATASET_ID', \
                                                       str(dataset_id)).replace('START', str(start)).replace('END', str(end)))
        for info in cur_data_info:
            info['dataset_id'] = dataset_id
            data_info.append(info)

    if len(data_info) < total_num:
        print(f'Warning: total number of prompts is less than {total_num}')
        total_num = len(data_info)
        random_data_info = data_info
    else:
        random_data_info = random.sample(data_info, total_num)
        # sort by dataset_id
        random_data_info = sorted(random_data_info, key = lambda x: x['dataset_id'])

    avg_prompt_len = 0
    avg_turn = 0
    avg_context_len = 0
    for info in random_data_info:
        avg_prompt_len += info['start']
        avg_turn += info['turn']
        avg_context_len += info['end'] - info['start']
    summary_random_info['avg_prompt_len'] = avg_prompt_len / total_num
    summary_random_info['avg_turn'] = avg_turn / total_num + 1
    summary_random_info['avg_context_len'] = avg_context_len / total_num
    summary_random_info['total_num'] = total_num
    summary_random_info['random_data_info'] = random_data_info

    with open(f'./data/random_prompts/random_prompt_{start}_{end}_summary.json', 'w') as f:
        json.dump(summary_random_info, f, indent=2)

    # collect prompts
    prev_dataset_id = -1
    for info in random_data_info:
        if info['dataset_id'] != prev_dataset_id:
            dataset_path = f'./data/ultrachat_{info["dataset_id"]}.jsonl'
            cur_dataset = load_jsonl(dataset_path)
            prev_dataset_id = info['dataset_id']
        turn = info['turn']
        prompt = cur_dataset[info['idx']]['data'][:(turn + 1) * 2]
        prompts.append(prompt)
        # jsonl dump line by line
        with open(f'./data/random_prompts/random_prompt_{start}_{end}.jsonl', 'a') as f:
            f.write(json.dumps(prompt) + '\n')
def main():
    filter_path = './data/filtered_info/prompt_DATASET_ID_len_START_END.json'
    dataset_id_begin = 0
    dataset_id_end = 9
    start = 1000
    end = 3000
    step = 500
    total_step = (end - start) // step + 1
    for i in range(total_step):
        cur_start = start + i * step
        cur_end = cur_start + step if i < total_step - 1 else 100000
        collect_random_prompts(filter_path, dataset_id_begin, dataset_id_end, cur_start, cur_end)

if __name__ == '__main__':
    main()