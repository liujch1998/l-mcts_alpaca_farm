accelerate launch \
    --config_file accelerate.cfg \
    ../examples/rlhf_ppo.py \
    --wandb_project "l-MCTS" \
    --step_per_device_batch_size 2 --rollout_per_device_batch_size 32 --per_device_eval_batch_size 32 \
    --output_dir runs/rlhf_ppo \
    --reward_model_name_or_path "pretrained_models/reward-model-sim" \
    --policy_model_name_or_path "runs/rlhf_ppo/checkpoint-80" \
    --value_model_name_or_path "runs/rlhf_ppo/checkpoint-80/value" \
    --ref_policy_model_name_or_path "pretrained_models/sft10k" \
    --rollout_batch_size 512 --step_batch_size 256 \
    --learning_rate 1e-5 --warmup_steps 5 --total_epochs 10 --flash_attn True \
    --prompt_dict_path "../examples/prompts/v0_inputs_noinputs.json" --save_steps 20 \
    --mode eval \
    --report_value True --use_value_in_decoding False --run_name "ckpt80_greedy_n128" # greedy
    # --report_value True --use_value_in_decoding True --run_name "ckpt80_greedy-by-value_k2_n128" # greedy-by-value
    # --use_mcts True --run_name "ckpt80_mcts_k2_n128" --init_v_with_parent True --kl_coef 0.0067 --debug False --visualize False # mcts

accelerate launch \
    --config_file accelerate.cfg \
    ../examples/rlhf_ppo.py \
    --wandb_project "l-MCTS" \
    --step_per_device_batch_size 2 --rollout_per_device_batch_size 32 --per_device_eval_batch_size 32 \
    --output_dir runs/rlhf_ppo_no-whiten-rewards \
    --reward_model_name_or_path "pretrained_models/reward-model-sim" \
    --policy_model_name_or_path "runs/rlhf_ppo_no-whiten-rewards/checkpoint-80" \
    --value_model_name_or_path "runs/rlhf_ppo_no-whiten-rewards/checkpoint-80/value" \
    --ref_policy_model_name_or_path "pretrained_models/sft10k" \
    --rollout_batch_size 512 --step_batch_size 256 \
    --learning_rate 1e-5 --warmup_steps 5 --total_epochs 10 --flash_attn True \
    --prompt_dict_path "../examples/prompts/v0_inputs_noinputs.json" --save_steps 20 \
    --mode eval \
    --use_mcts True --run_name "ckpt80_mcts_k2_n1" --init_v_with_parent True --kl_coef 0.0 --debug True --visualize True # mcts
    # --report_value True --use_value_in_decoding False --run_name "ckpt80_greedy_n128" # greedy