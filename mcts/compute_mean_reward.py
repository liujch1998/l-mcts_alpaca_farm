import json
import sys
import scipy
import numpy as np
import csv

with open(f'runs/rlhf_ppo/eval_ckpt80_greedy_n128.json') as f:
# with open(f'runs/rlhf_ppo/eval_ckpt80_t1.0_n128.json') as f:
# with open(f'runs/rlhf_ppo/eval_ckpt80_t0.7_n128.json') as f:
# with open(f'runs/rlhf_ppo/eval_ckpt80_topp0.5_n128.json') as f:
# with open(f'runs/rlhf_ppo/eval_ckpt80_beam2_n128.json') as f:
    ds1 = json.load(f)
rewards_1 = [d['reward'] for d in ds1]
print(f'Greedy: n={len(rewards_1)}, mean={np.mean(rewards_1)}, std={np.std(rewards_1)}')

# with open(f'runs/rlhf_ppo/eval_ckpt80_mcts_n128.json') as f:
# with open(f'runs/rlhf_ppo/eval_ckpt80_mcts_eos_n128.json') as f:
# with open(f'runs/rlhf_ppo/eval_ckpt80_mcts_eos_ivwp_n128.json') as f:
# with open(f'runs/rlhf_ppo/eval_ckpt80_mcts_eos_ivwp_sim20_n128.json') as f:
with open(f'runs/rlhf_ppo/eval_ckpt80_greedy-by-value_k2_n128.json') as f:
    ds2 = json.load(f)
rewards_2 = [d['reward'] for d in ds2]
print(f'MCTS: n={len(rewards_2)}, mean={np.mean(rewards_2)}, std={np.std(rewards_2)}')

results = scipy.stats.ttest_rel(rewards_1, rewards_2, alternative='less') # alternative is opposed to the null 
print(results)

with open(f'runs/rlhf_ppo/greedy_vs_value.tsv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=['instruction', 'input', 'output_greedy', 'reward_greedy', 'output_value', 'reward_value'])
    writer.writeheader()
    for d1, d2 in zip(ds1, ds2):
        writer.writerow({
            'instruction': d1['instruction'],
            'input': d1['input'],
            'output_greedy': d1['rlhf_ppo_ckpt_0'],
            'reward_greedy': d1['reward'],
            'output_value': d2['rlhf_ppo_ckpt_0'],
            'reward_value': d2['reward'],
        })

suffix = sys.argv[1]

# with open(f'runs/rlhf_ppo_no-whiten-rewards/eval_ckpt80_{suffix}.json') as f:
with open(f'runs/rlhf_ppo/eval_ckpt80_{suffix}.json') as f:
    ds = json.load(f)
# ds = ds[:32]
rewards = [d['reward'] for d in ds]
reward = sum(rewards) / len(rewards)
print(len(rewards))
print(f'Average reward: {reward}')

greedy_has_largest_value = []
for item in ds:
    for record in item['record']:
        greedy_has_largest_value.append(1 if record['topk_values'][0] >= record['topk_values'][1] else 0)
print(f'Greedy has largest value: {sum(greedy_has_largest_value) / len(greedy_has_largest_value)}')
