from typing import Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from einops import rearrange, reduce

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_lcm import LCMScheduler
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.light_transformer_for_diffusion import LightTransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    scaled_timestep = timestep_scaling * timestep
    c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
    c_out = scaled_timestep / (scaled_timestep**2 + sigma_data**2) ** 0.5
    return c_skip, c_out

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]

def extract_into_tensor(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def predicted_origin(
        model_output: torch.Tensor,
        timesteps: torch.Tensor,
        sample: torch.Tensor,
        prediction_type: str,
        alphas: torch.Tensor,
        sigmas: torch.Tensor
) -> torch.Tensor:
    if prediction_type == "epsilon":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "sample":
        pred_x_0 = model_output
    elif prediction_type == "v_prediction":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(f"Prediction type {prediction_type} currently not supported.")

    return pred_x_0


class DiffusionLightTransformerHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDIMScheduler,
            scheduler: LCMScheduler,
            # task params
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None, # DDIM solver
            # image
            crop_shape=(76, 76),
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            # arch
            n_layer=8,
            n_cond_layers=0,
            n_head=4,
            n_emb=256,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            causal_attn=True,
            time_as_cond=True,
            obs_as_cond=True,
            pred_action_steps_only=False,
            num_inference_timesteps=4, # LCM inference steps
            use_consistency=True,
            # learning by prune
            groups = [[1, 2] for _ in range(4)], # remove 1 in each group with 2 blocks, [6, 8], [4, 8], [2, 8]
            importance_score_type = 'svdscore',
            fixed_importance_score = 1.0,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        # import pdb;pdb.set_trace()
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim if obs_as_cond else (obs_feature_dim + action_dim)
        output_dim = input_dim
        cond_dim = obs_feature_dim if obs_as_cond else 0

        model = LightTransformerForDiffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            n_cond_layers=n_cond_layers,
            groups = groups,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.scheduler = scheduler
        self.num_inference_timesteps = num_inference_timesteps
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.importance_score_type = importance_score_type
        self.fixed_importance_score = fixed_importance_score
        self.use_consistency = use_consistency
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.groups = groups


    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None,
            **kwargs
            ):
        model = self.model

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        if self.use_consistency:

            latents = trajectory * self.scheduler.init_noise_sigma
            # set timesteps
            self.scheduler.set_timesteps(self.num_inference_timesteps)
            timesteps = self.scheduler.timesteps.to(latents.device)

            # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
            # eta (η) is only used with the DDIMScheduler, and between [0, 1]
            extra_step_kwargs = {}

            for i, t in enumerate(timesteps):
                latent_model_input = latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = self.model(
                    sample=latent_model_input,
                    timestep=t,
                    cond=cond)

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            
            trajectory = latents
        else:
            # use ddpm scheduler
            # set step values
            self.noise_scheduler.set_timesteps(self.num_inference_steps)

            for t in self.noise_scheduler.timesteps:
                # 1. apply conditioning
                trajectory[condition_mask] = condition_data[condition_mask]

                # 2. predict model output
                model_output = model(trajectory, t, cond)

                # 3. compute previous image: x_t -> x_t-1
                trajectory = self.noise_scheduler.step(
                    model_output, t, trajectory, 
                    generator=generator,
                    **kwargs
                    ).prev_sample
            
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            cond = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, 
            transformer_weight_decay: float, 
            obs_encoder_weight_decay: float,
            gumbel_gates_lr_scale: float,
            learning_rate: float, 
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(
            weight_decay=transformer_weight_decay,
            ignore_keys=['gumbel_gates'])
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        optim_groups.append({
            "params": self.model.gumbel_gates.parameters(),
            "weight_decay": 0.0, # gumbel gates will not be weight decayed
            "lr": learning_rate * gumbel_gates_lr_scale
        })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def compute_loss(self, batch, teacher_model=None, ema_model=None, solver=None):
        # normalize input
        # import pdb;pdb.set_trace()
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        To = self.n_obs_steps

        # handle different ways of passing observation
        cond = None
        trajectory = nactions
        if self.obs_as_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            cond = nobs_features.reshape(batch_size, To, -1)
            if self.pred_action_steps_only:
                start = To - 1
                end = start + self.n_action_steps
                trajectory = nactions[:,start:end]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            trajectory = torch.cat([nactions, nobs_features], dim=-1).detach()

        # # generate inpainting mask
        # if self.pred_action_steps_only:
        #     condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        # else:
        #     condition_mask = self.mask_generator(trajectory.shape)

        # 1. Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        latents = trajectory

        # 2. Sample a random timestep for each image t_n ~ U[0, N - k - 1] without bias.
        topk = self.noise_scheduler.config.num_train_timesteps // self.num_inference_steps
        index = torch.randint(0, self.num_inference_steps, (batch_size,), device=trajectory.device).long()
        start_timesteps = solver.ddim_timesteps[index]
        timesteps = start_timesteps - topk
        timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)

        # 3. Get boundary scalings for start_timesteps and (end) timesteps.
        c_skip_start, c_out_start = scalings_for_boundary_conditions(start_timesteps)
        c_skip_start, c_out_start = [append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]]
        c_skip, c_out = scalings_for_boundary_conditions(timesteps)
        c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]

        noisy_model_input = self.noise_scheduler.add_noise(latents, noise, start_timesteps)

        # # compute loss mask
        # loss_mask = ~condition_mask

        # # apply conditioning
        # noisy_model_input[condition_mask] = trajectory[condition_mask]

        noise_pred = self.model(
            sample=noisy_model_input, 
            timestep=start_timesteps, 
            cond=cond)

        pred_x_0 = predicted_origin(
            noise_pred,
            start_timesteps,
            noisy_model_input,
            self.noise_scheduler.config.prediction_type,
            torch.sqrt(self.noise_scheduler.alphas_cumprod),
            torch.sqrt(1 - self.noise_scheduler.alphas_cumprod))
        
        model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0

        if teacher_model is not None:
            with torch.no_grad():
                # teacher model the raw dp model, which predict the noise
                cond_teacher_output = teacher_model(
                    sample=noisy_model_input, 
                    timestep=start_timesteps, 
                    cond=cond)

                cond_pred_x0 = predicted_origin(
                    cond_teacher_output,
                    start_timesteps,
                    noisy_model_input,
                    'epsilon', # TODO: check if this is correct
                    torch.sqrt(self.noise_scheduler.alphas_cumprod),
                    torch.sqrt(1 - self.noise_scheduler.alphas_cumprod))
                
                x_prev = solver.ddim_step(cond_pred_x0, cond_teacher_output, index)

            with torch.no_grad():
                target_noise_pred = ema_model(
                    x_prev.float(),
                    timesteps,
                    cond=cond)
                pred_x_0 = predicted_origin(
                    target_noise_pred,
                    timesteps,
                    x_prev,
                    self.noise_scheduler.config.prediction_type,
                    torch.sqrt(self.noise_scheduler.alphas_cumprod),
                    torch.sqrt(1 - self.noise_scheduler.alphas_cumprod))
                target = c_skip * x_prev + c_out * pred_x_0
        else:
            target = latents

        # compute loss
        # reconstruction loss
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        # loss = loss * loss_mask.type(loss.dtype)
        # loss = reduce(loss, 'b ... -> b (...)', 'mean')
        # loss = loss.mean()
        return loss 
