"""
Usage:
Training:
# original diffusion policy
    python train.py --config-name=train_diffusion_lowdim_workspace
    # robomimic benchmark
    # lift
    python train.py --config-dir=. --config-name=train_diffusion_transformer_hybrid_workspace.yaml policy.crop_shape=[76,76] logging.project=diffusion_policy_reproduce task.name=lift_image_abs hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_${dataset_type}'
    python train.py --config-dir=. --config-name=train_diffusion_transformer_hybrid_workspace.yaml policy.crop_shape=[76,76] task.dataset_type=mh logging.project=diffusion_policy_reproduce task.name=lift_image_abs hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_${dataset_type}'
    # can
    python train.py --config-dir=. --config-name=train_diffusion_transformer_hybrid_workspace.yaml policy.crop_shape=[76,76] logging.project=diffusion_policy_reproduce task=can_image_abs hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_${dataset_type}'
    python train.py --config-dir=. --config-name=train_diffusion_transformer_hybrid_workspace.yaml policy.crop_shape=[76,76] task.dataset_type=mh logging.project=diffusion_policy_reproduce task=can_image_abs hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_${dataset_type}'
    # square
    python train.py --config-dir=. --config-name=train_diffusion_transformer_hybrid_workspace.yaml policy.crop_shape=[76,76] logging.project=diffusion_policy_reproduce task=square_image_abs hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_${dataset_type}'
    python train.py --config-dir=. --config-name=train_diffusion_transformer_hybrid_workspace.yaml policy.crop_shape=[76,76] task.dataset_type=mh logging.project=diffusion_policy_reproduce task=square_image_abs hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_${dataset_type}'
    # transport
    python train.py --config-dir=. --config-name=train_diffusion_transformer_hybrid_workspace.yaml policy.crop_shape=[76,76] logging.project=diffusion_policy_reproduce task=transport_image_abs hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_${dataset_type}'
    python train.py --config-dir=. --config-name=train_diffusion_transformer_hybrid_workspace.yaml policy.crop_shape=[76,76] task.dataset_type=mh logging.project=diffusion_policy_reproduce task=square_image_abs hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_${dataset_type}'
    # toolhang
    python train.py --config-dir=. --config-name=train_diffusion_transformer_hybrid_workspace.yaml policy.crop_shape=[76,76] logging.project=diffusion_policy_reproduce task=tool_hang_image_abs hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
    # pusht benchmark
    python train.py --config-dir=. --config-name=train_diffusion_transformer_hybrid_workspace.yaml logging.project=diffusion_policy_reproduce task=pusht_image hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
    # kitchen benchmark
    python train.py --config-dir=. --config-name=train_diffusion_transformer_lowdim_kitchen_workspace.yaml logging.project=diffusion_policy_reproduce hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
    # block push benchmark
    python train.py --config-dir=. --config-name=train_diffusion_transformer_lowdim_workspace.yaml logging.project=diffusion_policy_reproduce hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

    ### multirun
    python ray_train_multirun.py --config-name=train_diffusion_transformer_hybrid_workspace.yaml --config-dir=diffusion_policy/config --seeds=42,43,44 --monitor_key=test/mean_score -- policy.crop_shape=[76,76] logging.project=diffusion_policy_reproduce task.name=square_image_abs multi_run.run_dir=data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_${dataset_type} multi_run.wandb_name_base=${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_${dataset_type}
    
# light transformer hybrid, prune by learning
    # pusht
    python train.py --config-dir=. --config-name=train_diffusion_light_transformer_hybrid_workspace.yaml "task=pusht_image" "training.pretrain.pretrained_model_path='data/experiments/image/pusht/diffusion_policy_transformer/train_0/checkpoints/epoch=0100-test_mean_score=0.748.ckpt'" "policy.groups=[[6, 8]]" "hydra.run.dir='data/outputs/prune_by_learning/prune_train_0_6_8/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'"

# TODO: add prune by score and prune by index
    # prune by score
    # pusht
    python train.py --config-dir=. --config-name=train_diffusion_light_transformer_hybrid_workspace_prune_by_score.yaml training.seed=42 training.device=cuda:0 task=pusht_image policy.pretrain.pretrained_model_path='data/experiments/image/pusht/diffusion_policy_transformer/train_1/checkpoints/epoch=0400-test_mean_score=0.817.ckpt' hydra.run.dir='data/outputs/prune_by_score/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
    python train.py --config-dir=. --config-name=train_diffusion_light_transformer_hybrid_workspace_prune_by_score.yaml training.seed=42 training.device=cuda:0 task=pusht_image policy.pretrain.pretrained_model_path='data/experiments/image/pusht/diffusion_policy_transformer/train_2/checkpoints/epoch=0150-test_mean_score=0.752.ckpt' hydra.run.dir='data/outputs/prune_by_score/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
    # prune by index
    # pusht
    python train.py --config-dir=. --config-name=train_diffusion_light_transformer_hybrid_workspace_prune_by_index.yaml training.seed=42 training.device=cuda:0 task=pusht_image hydra.run.dir='data/outputs/prune_by_index/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    # if cfg.training.debug:
    #     import pdb;pdb.set_trace()
    # use +output_dir=path or hydra.run.dir=path to set the output_dir in cfg, output_dir is set to HydraConfig.get().runtime.output_dir by default
    workspace: BaseWorkspace = cls(cfg, output_dir=cfg.get('output_dir'))
    workspace.run()

if __name__ == "__main__":
    main()
