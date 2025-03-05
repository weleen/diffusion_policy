if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_light_transformer_hybrid_image_policy import DiffusionLightTransformerHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

import logging
logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionLightTransformerHybridWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        logger.info(f"Initializing TrainDiffusionLightTransformerHybridWorkspace with config: {cfg}")

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionLightTransformerHybridImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionLightTransformerHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        if cfg.training.pretrain.pretrained_model_path:
            pretrained_model_path = pathlib.Path(cfg.training.pretrain.pretrained_model_path)
            assert pretrained_model_path.is_file(), f"Pretrained model {pretrained_model_path} does not exist, check the path!"
            print(f"Loading pretrained model from {pretrained_model_path} with "
                  f"include_keys = {cfg.training.pretrain.include_keys} and "
                  f"exclude_keys = {cfg.training.pretrain.exclude_keys}")
            import dill
            payload = torch.load(pretrained_model_path.open('rb'), pickle_module=dill)
            self.load_payload(payload,
                              exclude_keys=cfg.training.pretrain.exclude_keys,
                              include_keys=cfg.training.pretrain.include_keys,
                              strict=False)
            if cfg.training.pretrain.use_ema:
                # initialize self.model with self.ema_model.state_dict()
                self.model.load_state_dict(self.ema_model.state_dict())

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        if cfg.training.debug:
            cfg.task.env_runner.n_train = 2
            cfg.task.env_runner.n_train_vis = 1
            cfg.task.env_runner.n_test = 2
            cfg.task.env_runner.n_test_vis = 1
 
        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        
        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # Log pruning decision at the beginning
        if self.ema_model is not None:
            print(f"Initial model pruning structure:")
            model_to_log = self.ema_model if self.ema_model is not None else self.model
            layer_offset = 0
            initial_decisions = []
            initial_conf = []
            for gate, option in zip(model_to_log.model.gumbel_gates, model_to_log.model.options):
                # Move tensors to the right device if needed
                gate = gate.to(device)
                option = option.to(device)
                
                # Get the current selection
                selected = gate.max(1)[1].item()
                mask = option[selected]
                selected_layers = (mask.nonzero().cpu()+layer_offset).squeeze().tolist()
                if not isinstance(selected_layers, list):
                    selected_layers = [selected_layers]
                
                # Calculate confidence scores
                initial_conf.append(max(torch.softmax(gate*model_to_log.model.scaling, dim=1).detach().cpu().tolist()[0]))
                initial_decisions.extend(selected_layers)
                layer_offset += option.size(1)
            
            logger.info(f"Initial selection decisions: {initial_decisions}")
            logger.info(f"Initial selection confidence: {initial_conf}")
            
            # Also log to wandb if available
            if wandb.run is not None:
                wandb.run.summary["initial_selection_decisions"] = str(initial_decisions)
                wandb.run.summary["initial_selection_confidence"] = str(initial_conf)

        # get the pruned model
        pruned_model = self.save_pruned_model(os.path.join(self.output_dir, 'checkpoints', 'pruned_model_epoch_0.ckpt'), device=device)

        # run evaluation at the beggining
        # original model
        if self.ema_model is not None:
            model_to_eval = self.ema_model
            logger.info(f"Using EMA model for evaluation")
            model_to_eval.eval()
        else:
            model_to_eval = self.model
            logger.info(f"Using original model for evaluation")
        runner_log = env_runner.run(model_to_eval)
        # log all
        logger.info(f"\nOriginal model evaluation\n{runner_log}")

        # initial pruned model
        runner_log = env_runner.run(pruned_model)
        # log all
        logger.info(f"\nInitial pruned model evaluation\n{runner_log}")

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        total_steps = len(train_dataloader) * cfg.training.num_epochs

        # Start training!
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                
                # Update temperature and scaling parameters for Gumbel softmax
                tau = cfg.training.tau_range[0] + (cfg.training.tau_range[1] - cfg.training.tau_range[0]) * self.global_step / max(total_steps - 1, 1)
                scaling = cfg.training.scaling_range[0] + (cfg.training.scaling_range[1] - cfg.training.scaling_range[0]) * self.global_step / max(total_steps - 1, 1)
                
                self.model.model.tau = tau
                self.model.model.scaling = scaling
                if self.ema_model is not None:
                    self.ema_model.model.tau = tau
                    self.ema_model.model.scaling = scaling
                
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        # if cfg.training.debug:
                        #     import pdb;pdb.set_trace()
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0],
                            'tau': tau,
                            'scaling': scaling
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # Log pruning decisions
                if self.ema_model is not None:
                    model_to_log = self.ema_model
                else:
                    model_to_log = self.model

                layer_offset = 0
                decisions = []
                conf = []
                for gate, option in zip(model_to_log.model.gumbel_gates, model_to_log.model.options):
                    selected = gate.max(1)[1].item()
                    mask = option[selected]
                    selected_layers = (mask.nonzero().cpu()+layer_offset).squeeze().tolist()
                    if isinstance(selected_layers, int):
                        selected_layers = [selected_layers]
                    conf.append(max(torch.softmax(gate*model_to_log.model.scaling, dim=1).detach().cpu().tolist()[0]))
                    decisions.extend(selected_layers)
                    layer_offset += option.size(1)
                step_log['selection_decisions'] = str(decisions)
                step_log['selection_confidence'] = str(conf)
                print(f"Selection decisions: {decisions}")
                print(f"Selection confidence: {conf}")

                # ========= eval for this epoch ==========
                # policy = self.model
                # if cfg.training.use_ema:
                #     policy = self.ema_model
                # get the pruned model
                if (self.epoch % cfg.training.val_every) == 0:
                    save = False # do not save the model

                if (self.epoch % cfg.training.rollout_every) == 0 or \
                    (self.epoch % cfg.training.checkpoint_every) == 0:
                    save = True

                pruned_model = self.save_pruned_model(os.path.join(self.output_dir, 'checkpoints', f'pruned_model_epoch_{self.epoch}.ckpt'), device=device, save=save)

                policy = pruned_model # use the pruned model for evaluation
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = policy.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

                
    def save_pruned_model(self, output_path, device='cpu', save=True):
        """Save the pruned model based on the learned pruning decisions"""
        if self.ema_model is None:
            model = self.model
        else:
            model = self.ema_model

        # Create a deep copy of the model to avoid modifying the original
        pruned_model = copy.deepcopy(model)
        pruned_model.to(device)

        # Get pruning decisions
        layer_offset = 0
        kept_layers = []
        for gate, option in zip(pruned_model.model.gumbel_gates, pruned_model.model.options):
            selected = gate.max(1)[1].item()
            mask = option[selected]
            selected_layers = (mask.nonzero().cpu()+layer_offset).squeeze().tolist()
            if isinstance(selected_layers, int):
                selected_layers = [selected_layers]
            kept_layers.extend(selected_layers)
            layer_offset += option.size(1)

        print(f"Prune model with kept layers: {kept_layers}")

        # Remove pruned layers from the model based on the architecture
        # TODO: dirty hack, need to fix this
        new_blocks = []
        for i in range(model.model.decoder.num_layers):
            if i in kept_layers:
                new_blocks.append(model.model.decoder.layers[i])
        # assemble the TransformerDecoderLayer
        pruned_model.model.decoder.layers = torch.nn.ModuleList(new_blocks)
        pruned_model.model.decoder.num_layers = len(kept_layers)
        # remove the gumbel gates
        del pruned_model.model.gumbel_gates
        
        # Save the pruned model state dict
        if save:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save({
                'model': self.model.state_dict(),
                'ema_model': self.ema_model.state_dict(),
                'pruned_model': pruned_model.state_dict(),
                'kept_layers': kept_layers,
                'normalizer': self.model.normalizer.state_dict()
            }, output_path)
            print(f"Saved pruned model to {output_path}")
        # TODO: Remove pruned layers from the model based on the architecture
        return pruned_model

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionLightTransformerHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
