"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
# robomimic benchmark
# lift 
python train.py --config-dir=. --config-name=train_diffusion_transformer_hybrid_workspace.yaml task=lift_image_abs hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
python train.py --config-dir=. --config-name=train_diffusion_transformer_hybrid_workspace.yaml task=lift_image hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
# can
python train.py --config-dir=. --config-name=train_diffusion_transformer_hybrid_workspace.yaml task=can_image_abs hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
python train.py --config-dir=. --config-name=train_diffusion_transformer_hybrid_workspace.yaml task=can_image hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
# square
python train.py --config-dir=. --config-name=train_diffusion_transformer_hybrid_workspace.yaml task=square_image_abs hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
python train.py --config-dir=. --config-name=train_diffusion_transformer_hybrid_workspace.yaml task=square_image hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
# transport
python train.py --config-dir=. --config-name=train_diffusion_transformer_hybrid_workspace.yaml task=transport_image_abs hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
python train.py --config-dir=. --config-name=train_diffusion_transformer_hybrid_workspace.yaml task=transport_image hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
# toolhang
python train.py --config-dir=. --config-name=train_diffusion_transformer_hybrid_workspace.yaml task=tool_hang_image_abs hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
python train.py --config-dir=. --config-name=train_diffusion_transformer_hybrid_workspace.yaml task=tool_hang_image hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
# pusht benchmark
# pusht
python train.py --config-dir=. --config-name=train_diffusion_transformer_hybrid_workspace.yaml task=pusht_image hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
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
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
