python train.py --config-dir=.  \
  --config-name=image_pusht_diffusion_policy_cnn.yaml  \
  training.seed=42  \
  training.device=cuda:0  \
  hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'