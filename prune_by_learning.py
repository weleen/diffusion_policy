"""
Usage:
python prune_by_learning.py --delta-w --lora --config-dir=. --config-name=train_diffusion_transformer_hybrid_workspace.yaml task=pusht_image hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import click
import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('--delta-w', is_flag=True)
@click.option('--lora', is_flag=True)
@click.option('--lora-rank', type=int, default=16)
@click.option('--scaling-range', nargs='+', type=float, default=[1e2, 1e2]) # this controls the one-hotness of the learnable probability, see https://arxiv.org/abs/2409.17481
@click.option('--tau-range', nargs='+', type=float, default=[4, 0.1]) # this controls the temperature of the gumbel softmax.
@click.option('--config-dir', type=str, default='.')
@click.option('--config-name', type=str, default='train_diffusion_transformer_hybrid_workspace.yaml')
@click.option('--task', type=str, default='pusht_image')
@click.option('--hydra-run-dir', type=str, default='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}')
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    """
    This script is used to prune the model by learning the mask to prune the blocks.
    """
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
