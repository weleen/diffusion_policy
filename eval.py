"""
Usage:
python eval.py --checkpoint data/experiments/image/pusht/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt -o data/eval/diffusion_policy_transformer_hybrid_pusht_train_0_latest
python eval.py --checkpoint data/experiments/image/pusht/diffusion_policy_transformer/train_0/checkpoints/epoch=0100-test_mean_score=0.748.ckpt -o data/eval/diffusion_policy_transformer_hybrid_pusht_train_0_epoch_100
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('-e', '--eval-time', is_flag=True, default=False)
def main(checkpoint, output_dir, device, eval_time):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # register hooks for checking inference time and memory usage
    if eval_time:
        # count parameters of the model
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters())

        submodules = {
            # diffusion transformer
            "policy.model": policy.model,
            "policy.model.input_emb": policy.model.input_emb,
            "policy.model.cond_obs_emb": policy.model.cond_obs_emb,
            "policy.model.encoder": policy.model.encoder,
            "policy.model.decoder": policy.model.decoder,
            "policy.model.ln_f": policy.model.ln_f,
            "policy.model.head": policy.model.head,
            # obs encoder
            "policy.obs_encoder": policy.obs_encoder,
        }

        for name, submodule in submodules.items():
            num_params = count_parameters(submodule)
            print(f"{name} - Number of parameters: {num_params / 1e6} M")

        from lightvla.hooks import start_time_setup_hook, execute_time_record_hook, execution_times, parameters_table, write_to_excel, print_table
        from functools import partial
        hook_handles = []
        for name, submodule in submodules.items():
            print(f"registering execute time hooks for {name}")
            hook_handles.append(submodule.register_forward_pre_hook(start_time_setup_hook))
            hook_handles.append(submodule.register_forward_hook(partial(execute_time_record_hook, name)))

    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
