#!/bin/bash

# pusht
# train_1
# DP-T-D6-6-8
python train.py --config-dir=. --config-name=train_diffusion_light_transformer_hybrid_workspace.yaml "task=pusht_image" "training.pretrain.pretrained_model_path='data/experiments/image/pusht/diffusion_policy_transformer/train_1/checkpoints/epoch=0400-test_mean_score=0.817.ckpt'" "policy.groups=[[6, 8]]" "hydra.run.dir='data/outputs/prune_by_learning/prune_train_1/DP-T-D6-6-8/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'" "training.device=cuda:0" &
# DP-T-D6-3-4
python train.py --config-dir=. --config-name=train_diffusion_light_transformer_hybrid_workspace.yaml "task=pusht_image" "training.pretrain.pretrained_model_path='data/experiments/image/pusht/diffusion_policy_transformer/train_1/checkpoints/epoch=0400-test_mean_score=0.817.ckpt'" "policy.groups=[[3, 4], [3, 4]]" "hydra.run.dir='data/outputs/prune_by_learning/prune_train_1/DP-T-D6-3-4/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'" "training.device=cuda:1" &
# DP-T-D4-4-8
python train.py --config-dir=. --config-name=train_diffusion_light_transformer_hybrid_workspace.yaml "task=pusht_image" "training.pretrain.pretrained_model_path='data/experiments/image/pusht/diffusion_policy_transformer/train_1/checkpoints/epoch=0400-test_mean_score=0.817.ckpt'" "policy.groups=[[4, 8]]" "hydra.run.dir='data/outputs/prune_by_learning/prune_train_1/DP-T-D4-4-8/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'" "training.device=cuda:2" &
# DP-T-D4-2-4
python train.py --config-dir=. --config-name=train_diffusion_light_transformer_hybrid_workspace.yaml "task=pusht_image" "training.pretrain.pretrained_model_path='data/experiments/image/pusht/diffusion_policy_transformer/train_1/checkpoints/epoch=0400-test_mean_score=0.817.ckpt'" "policy.groups=[[2, 4], [2, 4]]" "hydra.run.dir='data/outputs/prune_by_learning/prune_train_1/DP-T-D4-2-4/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'" "training.device=cuda:3" &
# DP-T-D4-1-2
python train.py --config-dir=. --config-name=train_diffusion_light_transformer_hybrid_workspace.yaml "task=pusht_image" "training.pretrain.pretrained_model_path='data/experiments/image/pusht/diffusion_policy_transformer/train_1/checkpoints/epoch=0400-test_mean_score=0.817.ckpt'" "policy.groups=[[1, 2], [1, 2], [1, 2], [1, 2]]" "hydra.run.dir='data/outputs/prune_by_learning/prune_train_1/DP-T-D4-1-2/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'" "training.device=cuda:4" &
# DP-T-D2-2-8
python train.py --config-dir=. --config-name=train_diffusion_light_transformer_hybrid_workspace.yaml "task=pusht_image" "training.pretrain.pretrained_model_path='data/experiments/image/pusht/diffusion_policy_transformer/train_1/checkpoints/epoch=0400-test_mean_score=0.817.ckpt'" "policy.groups=[[2, 8]]" "hydra.run.dir='data/outputs/prune_by_learning/prune_train_1/DP-T-D2-2-8/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'" "training.device=cuda:5" &
# DP-T-D2-1-4
python train.py --config-dir=. --config-name=train_diffusion_light_transformer_hybrid_workspace.yaml "task=pusht_image" "training.pretrain.pretrained_model_path='data/experiments/image/pusht/diffusion_policy_transformer/train_1/checkpoints/epoch=0400-test_mean_score=0.817.ckpt'" "policy.groups=[[1, 4], [1, 4]]" "hydra.run.dir='data/outputs/prune_by_learning/prune_train_1/DP-T-D2-1-4/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'" "training.device=cuda:6" &
