import matplotlib.pyplot as plt
import numpy as np
import json
import matplotlib.lines as mlines

def draw_ablation(features, steps, avg_prompt_len, avg_turn, avg_context_len, total_num, save_path):
    plt.figure(figsize=(10, 5))
    for i in range(steps):
        plt.plot(features[:, i], label=f'window {i}')

    plt.ylim(0, 1.1)
    plt.grid()

    custom_entries = [
        mlines.Line2D([], [], color='none', marker='None', linestyle='None', label=f'Avg Prompt Len: {avg_prompt_len}'),
        mlines.Line2D([], [], color='none', marker='None', linestyle='None', label=f'Avg Turn: {avg_turn}'),
        mlines.Line2D([], [], color='none', marker='None', linestyle='None', label=f'Avg Context Len: {avg_context_len}'),
        mlines.Line2D([], [], color='none', marker='None', linestyle='None', label=f'Total Num: {total_num}')
    ]

    handles, labels = plt.gca().get_legend_handles_labels()

    # Combine existing handles (if any) with custom ones
    handles.extend(custom_entries)

    # Create the legend with the combined handles
    # plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    plt.legend(handles=handles)
    plt.title('Hit rates for different windows')
    plt.xlabel('Layer')
    plt.ylabel('Hit rate (%)')
    plt.tight_layout()  # Adjust layout to make room for the legend

    plt.savefig(save_path)

def main():
    feature_paths = [
        './data/features_finegrained/features_1000_1500_step_128.jsonl',
        './data/features_finegrained/features_1500_2000_step_128.jsonl',
        './data/features_finegrained/features_2000_2500_step_128.jsonl',
        './data/features_finegrained/features_2500_3000_step_128.jsonl',
        './data/features_finegrained/features_3000_100000_step_128.jsonl',
    ]

    data_paths = [
        './data/random_prompts/random_prompt_1000_1500_summary.json',
        './data/random_prompts/random_prompt_1500_2000_summary.json',
        './data/random_prompts/random_prompt_2000_2500_summary.json',
        './data/random_prompts/random_prompt_2500_3000_summary.json',
        './data/random_prompts/random_prompt_3000_100000_summary.json',
    ]

    layers = 32
    steps = 4

    for i in range(len(feature_paths)):
        with open(feature_paths[i], 'r') as f:
            features = np.array([json.loads(line) for line in f])
            features = np.array(features).reshape(-1, layers, steps).mean(axis=0)
        print(features.shape)
        with open(data_paths[i], 'r') as f:
            data = json.load(f)

        draw_ablation(features, steps, data['avg_prompt_len'], data['avg_turn'], data['avg_context_len'], data['total_num'], f'./data/figures/ablation_finegrained_{i}.png')

if __name__ == '__main__':
    main()