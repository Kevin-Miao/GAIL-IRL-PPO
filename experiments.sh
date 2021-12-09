python train_imitation.py --algo airl --cuda --env_id BreakoutNoFrameskip-v4     --buffer /notebooks/rl-baselines3-zoo/buffers/BreakoutNoFrameskip-v4/size10000_std0.0_prand0.0.pth  --num_steps 1000000 --eval_interval 5000 --rollout_length 5000 --seed 0 --discrete

python train_imitation.py --algo airl --cuda --env_id BreakoutNoFrameskip-v4     --buffer /notebooks/rl-baselines3-zoo/buffers/BreakoutNoFrameskip-v4/size10000_std0.0_prand0.05.pth  --num_steps 1000000 --eval_interval 5000 --rollout_length 5000 --seed 0 --discrete

python train_imitation.py --algo airl --cuda --env_id BreakoutNoFrameskip-v4     --buffer /notebooks/rl-baselines3-zoo/buffers/BreakoutNoFrameskip-v4/size10000_std0.0_prand0.2.pth  --num_steps 1000000 --eval_interval 5000 --rollout_length 5000 --seed 0 --discrete

python train_imitation.py --algo airl --cuda --env_id BreakoutNoFrameskip-v4     --buffer /notebooks/rl-baselines3-zoo/buffers/BreakoutNoFrameskip-v4/size10000_std0.0_prand0.5.pth  --num_steps 1000000 --eval_interval 5000 --rollout_length 5000 --seed 0 --discrete
