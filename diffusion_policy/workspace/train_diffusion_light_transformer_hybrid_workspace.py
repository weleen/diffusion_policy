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
from typing import List, Generator
from itertools import accumulate
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


def extract_into_tensor(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class DDIMSolver:
    def __init__(self, alpha_cumprods: np.ndarray, timesteps: int = 1000, ddim_timesteps: int = 50) -> None:
        # DDIM sampling parameters
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        # convert to torch tensors
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

    def to(self, device: torch.device) -> "DDIMSolver":
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        return self

    def ddim_step(self, pred_x0: torch.Tensor, pred_noise: torch.Tensor,
                  timestep_index: torch.Tensor) -> torch.Tensor:
        alpha_cumprod_prev = extract_into_tensor(self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev

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
        assert cfg.training.use_ema, "set use_ema to True to use EMA"
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
        self.raw_model = copy.deepcopy(self.model) # this is the original model, used as a teacher model
    
        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        self.raw_model.eval()
        self.raw_model.requires_grad_(False)
        self.raw_model.to(device)

        noise_scheduler = self.model.noise_scheduler
        alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
        sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)

        solver = DDIMSolver(
            noise_scheduler.alphas_cumprod.numpy(),
            timesteps=noise_scheduler.config.num_train_timesteps,
            ddim_timesteps=cfg.policy.num_inference_steps,
        )

        encoder = self.model.obs_encoder
        teacher_model = self.ema_model.model

        encoder.requires_grad_(False)
        teacher_model.requires_grad_(False)
        # Also move the alpha and sigma noise schedules to device
        alpha_schedule = alpha_schedule.to(device)
        sigma_schedule = sigma_schedule.to(device)
        solver = solver.to(device)
        
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

        # calculate importance score
        if self.ema_model is not None:
            model_for_score = self.ema_model
        else:
            model_for_score = self.model

        def calculate_biscore(model, data_loader):
            # register hooks to the model
            @torch.no_grad()
            def compute_BI_score(module, inputs, outputs):
                # N, L, D 
                out_state = outputs
                in_state = inputs[0]
                cosine_sim = torch.nn.functional.cosine_similarity(out_state, in_state, dim=-1).mean().cpu()
                if hasattr(module, 'BI_score'):
                    module.BI_score.append(cosine_sim)
                else:
                    module.BI_score = [cosine_sim]
            hooks = []
            for layer in model.model.decoder.layers:
                hooks.append(layer.register_forward_hook(compute_BI_score))
            # iterate over the data loader
            for batch in data_loader:
                # device transfer
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                # forward pass
                model.compute_loss(batch)
                if cfg.training.debug:
                    break

            # collect scores
            layer_scores = []
            for layer in model.model.decoder.layers:
                layer_scores.append( sum(layer.BI_score) / len(layer.BI_score) )
            layer_scores = torch.tensor(layer_scores)
            # remove hooks
            for hook in hooks:
                hook.remove()
            # topk layers are less important
            return layer_scores

        def calculate_svdscore(model):
            @torch.no_grad()
            def compute_svd_diff(weight, partial_topk=0.5, topk=128, energy_threshold=0.9): # 0.95 for recoverability
                # compute the svd of the weight
                u, s, v = torch.svd(weight)
                # keep the topk singular values based on the energy
                energy = torch.sum(s**2)
                energy_cumsum = torch.cumsum(s**2, dim=0)
                topk_idx = torch.sum(energy_cumsum < energy * energy_threshold)
                # get the topk singular values
                topk = int(min(partial_topk * s.size(0), topk_idx, topk))
                topk_s = s.topk(topk).values
                # fill the rest with 0
                new_s = torch.zeros_like(s)
                new_s[:topk] = topk_s
                # compute the difference between the svd and the weight
                diff = torch.norm(weight - u @ torch.diag(new_s) @ v.t())
                return diff

            # Directly access transformer decoder layers
            scores = []
            for layer in model.model.decoder.layers:
                assert isinstance(layer, torch.nn.TransformerDecoderLayer)
                # get the parameters of the self_attn, multihead_attn, and ffn
                self_attn = layer.self_attn
                multihead_attn = layer.multihead_attn
                to_q, to_k, to_v = self_attn.in_proj_weight.chunk(3, dim=0)
                # compute SVD differences for self attention
                diff_sa_q = compute_svd_diff(to_q)
                diff_sa_k = compute_svd_diff(to_k)
                diff_sa_v = compute_svd_diff(to_v)
                to_q, to_k, to_v = multihead_attn.in_proj_weight.chunk(3, dim=0)
                # compute SVD differences for multihead attention
                diff_ma_q = compute_svd_diff(to_q)
                diff_ma_k = compute_svd_diff(to_k)
                diff_ma_v = compute_svd_diff(to_v)
                
                # compute SVD differences for feed forward network
                diff_ffn1 = compute_svd_diff(layer.linear1.weight)
                diff_ffn2 = compute_svd_diff(layer.linear2.weight)
                
                # sum up all differences for this layer
                layer_score = (diff_sa_q + diff_sa_k + diff_sa_v + 
                             diff_ma_q + diff_ma_k + diff_ma_v + 
                             diff_ffn1 + diff_ffn2).item()
                scores.append(layer_score)
            layer_scores = torch.tensor(scores)
            return layer_scores

        def calculate_importance_score(model, data_loader, importance_score_type, fixed_importance_score):
            if importance_score_type == "biscore":
                # calculate biscore
                importance_score = calculate_biscore(
                    model, data_loader)
                # importance_score = torch.ones(model.model.decoder.num_layers)
            elif importance_score_type == "svdscore":
                # calculate svdscore
                importance_score = calculate_svdscore(
                    model)
            elif importance_score_type == "fixed":
                # use fixed importance score
                importance_score = fixed_importance_score * torch.ones(model.model.decoder.num_layers)
            elif importance_score_type == "random":
                # use random importance score
                importance_score = torch.rand(model.model.decoder.num_layers)
            else:
                raise ValueError(f"Invalid importance score type: {cfg.policy.importance_score_type}")
            return importance_score

        # calculate importance score
        # for type in ["biscore", "svdscore", "fixed", "random"]:
        importance_score: torch.Tensor = calculate_importance_score(
            model_for_score,
            train_dataloader,
            cfg.policy.importance_score_type,
            cfg.policy.fixed_importance_score)
        print(f"Importance score type: {cfg.policy.importance_score_type}, score: {importance_score.cpu().tolist()}")

        # reset the gumbel_gates
        gates = self.model.model.gumbel_gates
        options = self.model.model.options
        # for each gate, set the value based on the importance score
        new_gates = copy.deepcopy(gates)
        # normalize the importance score
        importance_score = importance_score / importance_score.sum()
        groups = self.model.groups
        cum_groups = [0] + list(accumulate(group[1] for group in groups))
        for i, gate, option in zip(range(len(gates)), gates, options):
            # print(f"gate: {gate}, option: {option}")
            for j in range(len(option)):
                # get the probability of each option
                prob = (option[j] * importance_score[cum_groups[i]:cum_groups[i+1]]).sum()
                # print(f"prob: {prob}")
                new_gates[i][0][j].data.copy_(prob)
        # print(f"gates: {gates}\n")
        # print(f"new_gates: {new_gates}\n")
        self.model.model.gumbel_gates = new_gates

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
                        # pass the denoiser of the raw model as a teacher model, and the denoiser of the ema model as a second teacher model.
                        # the obs_encoder of the ema_model will not be ema updated.
                        train_loss = self.model.compute_loss(batch, self.raw_model.model, self.ema_model.model, solver)
                        loss = train_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)
                            # reset encoder, no ema for encoder
                            self.ema_model.obs_encoder = copy.deepcopy(self.model.obs_encoder)

                        # logging
                        train_loss_cpu = train_loss.item()
                        tepoch.set_postfix(loss=train_loss_cpu, refresh=False)
                        train_losses.append(train_loss_cpu)
                        step_log = {
                            'train_loss': train_loss_cpu,
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
                                loss = policy.compute_loss(batch, self.raw_model.model, self.ema_model.model, solver)
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
