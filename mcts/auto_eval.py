import json
import sys
import scipy
import numpy as np
import csv

with open(f'runs/rlhf_ppo/eval_ckpt80_greedy_n128.json') as f:
    ds1 = json.load(f)

with open(f'runs/rlhf_ppo/eval_ckpt80_mcts_eos_ivwp_sim20_n128.json') as f:
    ds2 = json.load(f)

# ds = []
# for (d1, d2) in zip(ds1, ds2):
#     ds.append({
#         'instruction': d1['instruction'],
#         'input': d1['input'],
#         'output_1': d1['rlhf_ppo_ckpt_0'],
#         'output_2': d2['rlhf_ppo_ckpt_0'],
#     })
# ds = ds[:1]

# from alpaca_farm.auto_annotations import PairwiseAutoAnnotator
# annotator = PairwiseAutoAnnotator()
# annotated = annotator.annotate_pairs(ds)
# print(annotated[0])

ds1 = [{'instruction': d['instruction'], 'input': d['input'], 'output': d['rlhf_ppo_ckpt_0']} for d in ds1]
ds2 = [{'instruction': d['instruction'], 'input': d['input'], 'output': d['output']} for d in ds2]

with open('ds1.json', 'w') as f:
    json.dump(ds1, f, indent=2)
with open('ds2.json', 'w') as f:
    json.dump(ds2, f, indent=2)
