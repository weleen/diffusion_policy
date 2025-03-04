#!/usr/bin/env python
"""
Converts DiffusionTransformerHybridImagePolicy PyTorch model to CoreML format.
"""
import os
from re import I
import sys
import logging
import glob
import gc
import time

import numpy as np
import einops
import shutil
import click
import dill
import hydra
import diffusers
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

import robomimic.utils.tensor_utils as TensorUtils

# Add the parent directory to the path to import mdt modules
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
print(f"sys.path: {sys.path}")

from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import DiffusionTransformerHybridImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

# Set up logging, store at ./mlpackage/torch2coreml.log
os.makedirs('./mlpackage', exist_ok=True)
logging.basicConfig(filename='./mlpackage/torch2coreml.log', level=logging.INFO)
logger = logging.getLogger(__name__)

# Version check for torch
try:
    import torch
    import packaging.version

    torch_version = packaging.version.parse(torch.__version__)
    min_torch_version = packaging.version.parse("2.1.0")

    if torch_version < min_torch_version:
        logger.warning(
            f"Warning: CoreMLTools optimize requires PyTorch {min_torch_version} or newer. "
            f"Found PyTorch {torch_version}. Some optimizations may not be available.\n"
            f"Consider upgrading with: pip install torch>=2.1.0"
        )
        HAS_REQUIRED_TORCH = False
    else:
        HAS_REQUIRED_TORCH = True
except ImportError:
    logger.error("PyTorch is not installed. Please install PyTorch 2.1.0 or newer.")
    sys.exit(1)

# Import coremltools with proper error handling
try:
    import coremltools as ct
    from coremltools.converters.mil import register_torch_op
    from coremltools.converters.mil.frontend.torch.ops import _get_inputs
except ImportError:
    logger.error(
        "CoreMLTools is not installed. Please install with: pip install coremltools"
    )
    sys.exit(1)


def _get_out_path(output_dir, submodule_name):
    fname = f"DP_{submodule_name}.mlpackage"
    return os.path.join(output_dir, fname)


def load_pretrained_model(
        checkpoint_path: str, 
        output_dir: str,
        n_layer: int,
        num_inference_steps: int,
        strict_loading: bool,
    ) -> BaseImagePolicy:
    """
    Load a pretrained DiffusionTransformerHybridImagePolicy model from the specified directory.

    Args:
        checkpoint_path: Path to the checkpoint file
        output_dir: Directory to save the output model

    Returns:
        The loaded DiffusionTransformerHybridImagePolicy model in evaluation mode
    """
    # load the checkpoint path
    assert (
        os.path.exists(checkpoint_path)
    ), f"Checkpoint file {checkpoint_path} does not exist"
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    if n_layer == cfg.policy.n_layer:
        logger.info(f"Loading model with {cfg.policy.n_layer} layers")
    else:
        cfg.policy.n_layer = n_layer
        logger.info(f"Overriding model with {n_layer} layers")
    if num_inference_steps == cfg.policy.num_inference_steps:
        logger.info(f"Loading model with {cfg.policy.num_inference_steps} inference steps")
    else:
        cfg.policy.num_inference_steps = num_inference_steps
        logger.info(f"Overriding model with {num_inference_steps} inference steps")

    workspace: BaseWorkspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload, exclude_keys=['optimizer'], include_keys=None, strict=strict_loading)

    # get policy from workspace
    logger.info(f"Loading model from {checkpoint_path}")
    policy: BaseImagePolicy = workspace.model
    if cfg.training.use_ema:
        policy: BaseImagePolicy = workspace.ema_model
    logger.info(f"Finished loading model {checkpoint_path}")
    policy.eval()  # Set to evaluation mode
    return policy


def _get_coreml_inputs(sample_inputs):
    return [
        ct.TensorType(
            name=k,
            shape=v.shape,
            dtype=v.numpy().dtype if isinstance(v, torch.Tensor) else v.dtype,
        )
        for k, v in sample_inputs.items()
    ]


def convert_to_coreml(
    submodule_name: Optional[str],
    torchscript_module: torch.nn.Module,
    sample_inputs: dict,
    output_names: List[str],
    output_dir: str = None,
    output_path: str = None,
    compute_unit: str = None,
    precision: str = None,
    check_output_correctness: bool = False,
) -> str:
    """
    Convert a PyTorch model to CoreML format.

    Args:
        submodule_name: Name of the submodule to convert
        torchscript_module: The PyTorch model to convert
        sample_inputs: Dictionary mapping input names to their shapes
        output_names: List of output names
        output_dir: Directory to save output model
        output_path: Path to save the CoreML model
        compute_unit: Compute unit to use (ALL, CPU_ONLY, etc.)
        precision: Precision to use (FLOAT32, FLOAT16, etc.)
        check_output_correctness: Whether to check the output correctness
    Returns:
        Path to the saved CoreML model
    """

    if output_path is None:
        output_path = _get_out_path(output_dir, submodule_name)

    compute_unit = ct.ComputeUnit[compute_unit]

    if os.path.exists(output_path):
        logger.info(f"Model already exists at {output_path}. Skipping conversion.")
        logger.info(f"Loading model from {output_path}")
        start = time.time()
        # Note: Note that each model load will trigger a model compilation which takes up to a few minutes.
        # The Swifty CLI we provide uses precompiled Core ML models (.mlmodelc) which incurs compilation only
        # upon first load and mitigates the load time in subsequent runs.
        coreml_model = ct.models.MLModel(output_path, compute_units=compute_unit)
        logger.info(f"Loading {output_path} took {time.time() - start:.1f} seconds")

        coreml_model.compute_unit = compute_unit
    else:
        logger.info(f"Converting {submodule_name} to CoreML...")
        coreml_model = ct.convert(
            torchscript_module,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS17,
            inputs=_get_coreml_inputs(sample_inputs),
            outputs=[
                ct.TensorType(name=name, dtype=np.float32) for name in output_names
            ],
            compute_units=compute_unit,
            compute_precision=precision,
            skip_model_load=not check_output_correctness,
        )

        # Save the model
        output_path = Path(output_path)
        os.makedirs(output_path.parent, exist_ok=True)

        try:
            coreml_model.save(str(output_path))
            logger.info(f"CoreML model saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    # Parity check PyTorch vs CoreML
    if check_output_correctness:
        baseline_out = torchscript_module(**sample_inputs).numpy()
        coreml_out = list(
            coreml_model.predict(
                {k: v.numpy() for k, v in sample_inputs.items()}
            ).values()
        )[0]
        np.testing.assert_allclose(
            baseline_out,
            coreml_out,
            rtol=1e-2, # 1e-3 may raise error
            atol=1e-2, # 1e-3 may raise error
            err_msg=f"assert allclose {submodule_name} baseline PyTorch to baseline CoreML failed",
        )

    del torchscript_module
    gc.collect()

    return coreml_model, output_path


def convert_obs_encoder_to_coreml(
    model: BaseImagePolicy,
    input_shapes: dict,
    output_dir: str,
    compute_unit: str = "ALL",
    check_output_correctness: bool = False,
) -> str:
    """
    Convert the obs encoder component of the BasePolicy to CoreML.
    /Users/wuyiming/opt/anaconda3/envs/robodiff/lib/python3.9/site-packages/robomimic/models/obs_nets.py
    Args:
        model: The BasePolicy model containing the obs encoder
        input_shapes: Dictionary of input shapes
        output_dir: Directory to save output model
        compute_unit: Compute unit to use (ALL, CPU_ONLY, etc.)
        check_output_correctness: Whether to check the output correctness

    Returns:
        Path to the saved CoreML model
    """
    logger.info("Converting obs encoder model to CoreML...")

    # Define output path
    output_path = _get_out_path(output_dir, "ObsEncoder")

    if os.path.exists(output_path):
        logger.info(
            f"`ObsEncoder` already exists at {output_path}, skipping conversion."
        )
        return

    # Extract obs encoder component from the BasePolicy
    obs_encoder = model.obs_encoder
    obs_encoder = obs_encoder.to(dtype=torch.float32)
    for param in obs_encoder.obs_nets['image'].nets.parameters():
        param.requires_grad = False

    # obs_encoder.eval() # not work
        
    # Define input shapes for obs encoder
    obs_inputs = {
        "agent_pos": torch.randn(input_shapes["agent_pos"], dtype=torch.float32).reshape(-1, *input_shapes["agent_pos"][2:]).clamp(-1, 1),  # (batch_size, 2, 2)
        "image": torch.randn(input_shapes["image"], dtype=torch.float32).reshape(-1, *input_shapes["image"][2:]).clamp(0, 1), # agent-view image, (0, 1)
    }
    obs_inputs_spec = {k: (v.shape, v.dtype) for k, v in obs_inputs.items()}
    logger.info(f"Obs encoder sample inputs spec: {obs_inputs_spec}")

    class ModelWrapper(torch.nn.Module):
        def __init__(self, obs_encoder, B, To, obs_input_keys):
            super().__init__()
            self.obs_encoder = obs_encoder
            self.B = B
            self.To = To
            self.obs_input_keys = obs_input_keys

        def forward(self, agent_pos, image):
            kwargs = {
                "agent_pos": agent_pos,
                "image": image,
            }
            feats = []
            for k in self.obs_input_keys:
                x = kwargs[k]
                if self.obs_encoder.obs_randomizers[k] is not None:
                    x = self.obs_encoder.obs_randomizers[k].forward_in(x) # fix __round__ error
                # maybe process with obs net
                if self.obs_encoder.obs_nets[k] is not None:
                    x = self.obs_encoder.obs_nets[k](x)
                    if self.obs_encoder.activation is not None:
                        x = self.obs_encoder.activation(x)
                # maybe process encoder output with randomizer
                if self.obs_encoder.obs_randomizers[k] is not None:
                    x = self.obs_encoder.obs_randomizers[k].forward_out(x)
                # flatten to [B, D]
                x = TensorUtils.flatten(x, begin_axis=1) # remove it is ok
                feats.append(x)
            return torch.cat(feats, dim=-1).reshape(self.B, self.To, -1)


    # Add debug prints before the conversion
    print("Agent pos shape:", obs_inputs["agent_pos"].shape)
    print("Image shape:", obs_inputs["image"].shape)

    # Also check the model's linear layer weights
    for name, param in obs_encoder.named_parameters():
        if 'weight' in name:
            print(f"{name} shape:", param.shape)

    reference_obs_encoder = ModelWrapper(
        obs_encoder, 
        input_shapes["n_obs_steps"], 
        input_shapes["obs_feature_dim"], 
        ["agent_pos", "image"]
    ).eval()
    logger.info(f"JIT tracing reference obs encoder...")
    traced_reference_obs_encoder = torch.jit.trace(
        reference_obs_encoder, 
        (obs_inputs["agent_pos"].to(torch.float32), 
        obs_inputs["image"].to(torch.float32))
    )
    logger.info(f"JIT tracing reference obs encoder done")

    # Convert to CoreML
    return convert_to_coreml(
        submodule_name="ObsEncoder",
        torchscript_module=traced_reference_obs_encoder,
        sample_inputs=obs_inputs,
        output_names=["cond"],
        output_dir=output_dir,
        compute_unit=compute_unit,
        check_output_correctness=check_output_correctness,
    )


def convert_diffusion_transformer_hybrid_image_policy_to_coreml(
    model: BaseImagePolicy,
    input_shapes: dict,
    output_dir: str,
    compute_unit: str = "ALL",
    check_output_correctness: bool = False,
) -> str:
    """
    Convert the diffusion transformer hybrid image policy model to CoreML.

    Args:
        model: The BaseImagePolicy model containing the diffusion transformer hybrid image policy
        input_shapes: Dictionary of input shapes
        output_dir: Directory to save output model
        compute_unit: Compute unit to use (ALL, CPU_ONLY, etc.)
        check_output_correctness: Whether to check the output correctness

    Returns:
        Path to the saved CoreML model
    """
    logger.info("Converting diffusion transformer hybrid image policy model to CoreML...")

    # Define output path
    output_path = _get_out_path(output_dir, "ImagePolicy")

    if os.path.exists(output_path):
        logger.info(
            f"`ImagePolicy` already exists at {output_path}, skipping conversion."
        )
        return

    # Extract diffusion transformer hybrid image policy component from the BaseImagePolicy
    policy = model.to(torch.float32)
    policy.eval()
    for param in policy.obs_encoder.obs_nets['image'].nets.parameters():
        param.requires_grad = False
    for param in policy.model.parameters():
        param.requires_grad = False
    print(f"sum(p.numel() for p in policy.model.parameters() if p.requires_grad): {sum(p.numel() for p in policy.model.parameters() if p.requires_grad)}")
 
    transformer = model.model  # TransformerForDiffusion
    transformer = transformer.to(torch.float32)
    for param in transformer.parameters():
        param.requires_grad = False
    print(f"sum(p.numel() for p in transformer.parameters() if p.requires_grad): {sum(p.numel() for p in transformer.parameters() if p.requires_grad)}")

    # Define input shapes for visual model
    bs = 1
    transformer_inputs = {
        "trajectory": torch.randn(
            bs, input_shapes["horizon"], input_shapes["action_dim"]),
        "cond_data": torch.zeros(
            size=(bs, input_shapes["horizon"], input_shapes["action_dim"]),
            dtype=torch.float32),
        "cond_mask": torch.zeros(
            size=(bs, input_shapes["horizon"], input_shapes["action_dim"]),
            dtype=torch.bool),
        "cond": torch.randn(
            bs, input_shapes["n_obs_steps"], input_shapes["obs_feature_dim"]),
    }

    transformer_inputs_spec = {k: (v.shape, v.dtype) for k, v in transformer_inputs.items()}
    logger.info(f"transformer sample inputs spec: {transformer_inputs_spec}")

    class TransformerModelWrapper(torch.nn.Module):
        def __init__(self, policy, Da):
            super().__init__()
            self.model = policy.model
            self.input_emb = self.model.input_emb
            self.cond_obs_emb = self.model.cond_obs_emb
            self.scheduler = policy.noise_scheduler
            self.scheduler.set_timesteps(policy.num_inference_steps)
            self.normalizer = policy.normalizer
            self.pred_action_steps_only = policy.pred_action_steps_only
            self.n_obs_steps = policy.n_obs_steps
            self.n_action_steps = policy.n_action_steps
            self.Da = Da
            self.n_dim = 256

        def forward_model(self, sample, timestep, cond):
            """
            x: (B,T,input_dim)
            timestep: (B,) or int, diffusion step
            cond: (B,T',cond_dim)
            output: (B,T,input_dim)
            """
            if not torch.is_tensor(timestep):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                timestep = torch.tensor([timestep], dtype=torch.long)
            elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
                timestep = timestep.unsqueeze(0)
            # # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timestep.expand(sample.shape[0])
            # SinusoidalPosEmb from examples/diffusion_policy/diffusion_policy/model/diffusion/positional_embedding.py
            time_emb = self.model.time_emb(timesteps).unsqueeze(1)
            # (B,1,n_emb)

            # process input
            b = sample.shape[0]
            # sample = einops.rearrange(sample, 'b to d -> (b to) d')
            inpu_emb = self.input_emb(sample)
            # inpu_emb = einops.rearrange(inpu_emb, '(b to) d -> b to d', b=b)
            # inpu_emb = torch.zeros((*(sample.shape[:-1]), 256)) # debug, comment out

            # encoder
            cond_embeddings = time_emb
            # if self.model.obs_as_cond: # True
            # cond = einops.rearrange(cond, 'b to d -> (b to) d')
            cond_obs_emb = self.cond_obs_emb(cond)
            # cond_obs_emb = einops.rearrange(cond_obs_emb, '(b to) d -> b to d', b=b)
            # cond_obs_emb = torch.zeros((*(cond.shape[:-1]), 256))
            # (B,To,n_emb)
            cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)

            tc = cond_embeddings.shape[1]
            position_embeddings = self.model.cond_pos_emb[:, :tc, :]  # each position maps to a (learnable) vector
            x = self.model.drop(cond_embeddings + position_embeddings)
            x = self.model.encoder(x)
            memory = x
            # (B,T_cond,n_emb)

            # x = torch.zeros((b, 3, 256)) # debug, comment out
            # memory = x
            
            # decoder
            token_embeddings = inpu_emb
            t = token_embeddings.shape[1]
            position_embeddings = self.model.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
            x = self.model.drop(token_embeddings + position_embeddings)
            # (B,T,n_emb)
            x = self.model.decoder(
                tgt=x,
                memory=memory,
                tgt_mask=self.model.mask,
                memory_mask=self.model.memory_mask
            )
            # (B,T,n_emb)

            # x = torch.zeros((b, 10, 256)) # debug, comment out

            # head
            x = self.model.ln_f(x)
            x = self.model.head(x)
            # (B,T,n_out)
            return x
        def scheduler_step(self, model_output, timestep, sample, generator=None, **kwargs):
            # return self.scheduler.step(model_output, t, trajectory, generator, return_dict, **kwargs).prev_sample

            # 1. compute alphas, betas
            t = timestep
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = self.scheduler.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0)
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev

            # 2. compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

            # 3. Clip "predicted x_0"
            if self.scheduler.clip_sample:
                pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

            # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * self.scheduler.betas[t]) / beta_prod_t
            current_sample_coeff = self.scheduler.alphas[t] ** (0.5) * beta_prod_t_prev / beta_prod_t

            # 5. Compute predicted previous sample µ_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

            # 6. Add noise
            variance = 0
            if t > 0:
                variance_noise = torch.randn(
                    model_output.shape, generator=generator, dtype=model_output.dtype
                )
                if self.scheduler.variance_type == "fixed_small_log":
                    variance = self.scheduler._get_variance(t, predicted_variance=None) * variance_noise
                else:
                    variance = (self.scheduler._get_variance(t, predicted_variance=None) ** 0.5) * variance_noise

            pred_prev_sample = pred_prev_sample + variance

            return pred_prev_sample
        def forward(self, trajectory, cond_data, cond_mask, cond):
            # # unwrapper nsample = self.policy.conditional_sample(cond_data, cond_mask, cond)
            for i, t in enumerate(self.scheduler.timesteps):
                print(f"ddpm step {i}")
                # 1. apply conditioning
                # cond_mask: (1,10,2), torch.bool
                # cond_data: (1,10,2), torch.float32
                # trajectory[cond_mask] = cond_data[cond_mask] # ERROR - converting 'index_put_' op, replace by torch.where
                trajectory = torch.where(cond_mask, cond_data, trajectory)

                # 2. predict model output
                # model_output = self.model(trajectory, t, cond) # ERROR - converting 'linear' op (located at: 'model/input_emb/input_emb.1'), unwrapping the function call
                # examples/diffusion_policy/diffusion_policy/model/diffusion/transformer_for_diffusion.py
                model_output = self.forward_model(trajectory, t, cond)
                # model_output = torch.zeros_like(trajectory) # debug, replace with zeros

                # 3. compute previous image: x_t -> x_t-1
                # trajectory = self.scheduler.step(
                #     model_output, t, trajectory,
                # ).prev_sample # ERROR - converting 'sub' op (located at: '61'):
                trajectory = self.scheduler_step(model_output, t, trajectory)

            # trajectory[cond_mask] = cond_data[cond_mask] # ERROR - converting 'index_put_' op, replace by torch.where
            trajectory = torch.where(cond_mask, cond_data, trajectory)
            nsample = trajectory
            # unnormalize prediction
            naction_pred = nsample[...,:self.Da]
            action_pred = self.normalizer['action'].unnormalize(naction_pred)

            # get action
            if self.pred_action_steps_only:
                action = action_pred
            else:
                start = self.n_obs_steps - 1
                end = start + self.n_action_steps
                action = action_pred[:,start:end]
            
            return action, action_pred

    # test the denoiser
    logger.info("Checking model dimensions...")
    test_denoiser = TransformerModelWrapper(policy, input_shapes["action_dim"]).eval()
    with torch.no_grad():
        test_out = test_denoiser(
            transformer_inputs["trajectory"],
            transformer_inputs["cond_data"],
            transformer_inputs["cond_mask"],
            transformer_inputs["cond"],
        )
        print(f"Test output shape: ({test_out[0].shape}, {test_out[1].shape})")
    logger.info("Checking model dimensions done.")
    # test the denoiser done.

    reference_transformer_model = TransformerModelWrapper(policy, input_shapes["action_dim"]).eval()
    logger.info(f"JIT tracing reference visual model...")
    traced_reference_visual_model = torch.jit.trace(
        reference_transformer_model, 
        (transformer_inputs["trajectory"].to(torch.float32),
         transformer_inputs["cond_data"].to(torch.float32),
         transformer_inputs["cond_mask"].to(torch.bool),
         transformer_inputs["cond"].to(torch.float32)),
        # check_trace=False
    )
    logger.info(f"JIT tracing reference visual model done")

    # Add debug prints before the conversion
    print("Trajectory shape:", transformer_inputs["trajectory"].shape)
    print("Cond data shape:", transformer_inputs["cond_data"].shape)
    print("Cond mask shape:", transformer_inputs["cond_mask"].shape)
    print("Cond shape:", transformer_inputs["cond"].shape)

    # Also check the model's linear layer weights
    for name, param in reference_transformer_model.named_parameters():
        if 'weight' in name:
            print(f"{name} shape:", param.shape)

    # Convert to CoreML
    return convert_to_coreml(
        submodule_name="ImagePolicy",
        torchscript_module=traced_reference_visual_model,
        sample_inputs=transformer_inputs,
        output_names=["action", "action_pred"],
        output_dir=output_dir,
        compute_unit=compute_unit,
        check_output_correctness=check_output_correctness,
    )


# python coreml/torch2coreml.py --checkpoint ../../pretrained_models/diffusion_policy/train_1/checkpoints/epoch=0400-test_mean_score=0.817.ckpt --num-inference-steps 10
# python coreml/torch2coreml.py --checkpoint ../../pretrained_models/diffusion_policy/train_1/checkpoints/epoch=0400-test_mean_score=0.817.ckpt --output-dir mlpackage/dp_d6_mlpackage --n-layer 6 --strict-loading False --num-inference-steps 10
@click.command()
@click.option("--checkpoint", type=str, required=True)
@click.option("--n-layer", type=int, default=8)
@click.option("--num-inference-steps", type=int, default=1000)
@click.option("--strict-loading", type=bool, default=True)
@click.option("--convert-obs-encoder", type=bool, default=True)
@click.option("--convert-model", type=bool, default=True)
@click.option("--output-dir", type=str, default="mlpackage/dp_mlpackage")
@click.option("--check-output-correctness", type=bool, default=False)
@click.option("--compute-unit", type=str, default="ALL")
@click.option("--clean-output-dir", type=bool, default=True)
@click.option("--bundle-resources", type=bool, default=False)
def main(checkpoint: str,
         n_layer: int,
         num_inference_steps: int,
         strict_loading: bool,
         output_dir: str,
         convert_obs_encoder: bool, 
         convert_model: bool, 
         check_output_correctness: bool, 
         compute_unit: str,
         clean_output_dir: bool,
         bundle_resources: bool):
    
    """Main function for converting DiffusionTransformerHybridImagePolicy components to CoreML models."""
    # Print version info
    logger.info(f"Using PyTorch {torch.__version__}, CoreMLTools {ct.__version__}")

    # Make output dir an absolute path
    output_dir = Path(output_dir).absolute()
    if clean_output_dir:
        if output_dir.exists():
            if (input(f"Output directory {output_dir} already exists. Continue? (Y/n)") or "Y") == "Y":
                shutil.rmtree(output_dir)
                logger.info(f"Output directory {output_dir} cleaned.")
            else:
                logger.info(f"Output directory {output_dir} not cleaned.")
                exit()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load the model using hydra config
    model = load_pretrained_model(checkpoint, output_dir, n_layer, num_inference_steps, strict_loading)

    # Get input shapes from config
    bs = 1
    input_shapes = {
        "image_size": 96,
        "agent_pos": (bs, 2, 2), # agent position
        "image": (bs, 2, 3, 96, 96), # agent-view image, (0, 1)
        "n_obs_steps": 2,
        "n_action_steps": 8,
        "action_dim": 2,
        "horizon": 10,
        "obs_feature_dim": 66
    }

    # Convert requested components
    converted_models = {}

    if convert_obs_encoder:
        logger.info("Converting Language Goal model to CoreML...")
        converted_models["obs_encoder"] = convert_obs_encoder_to_coreml(
            model=model,
            input_shapes=input_shapes,
            output_dir=output_dir,
            compute_unit=compute_unit,
            check_output_correctness=check_output_correctness,
        )
        logger.info(
            f"Obs encoder model converted and saved to {converted_models['obs_encoder']}"
        )

    if convert_model:
        logger.info("Converting Visual Goal model to CoreML...")
        converted_models["diffusion_transformer_hybrid_image_policy"] = convert_diffusion_transformer_hybrid_image_policy_to_coreml(
            model=model,
            input_shapes=input_shapes,
            output_dir=output_dir,
            compute_unit=compute_unit,
            check_output_correctness=check_output_correctness,
        )
        logger.info(
            f"Diffusion Transformer Hybrid Image Policy model converted and saved to {converted_models['diffusion_transformer_hybrid_image_policy']}"
        )

    # Bundle resources if requested
    if bundle_resources and converted_models:
        logger.info("Bundling resources for Swift CLI...")
        swift_resources_dir = os.path.join(output_dir, "DPResources")
        os.makedirs(swift_resources_dir, exist_ok=True)

        # Copy all converted models to the resources directory
        for component, model_path in converted_models.items():
            # Create symbolic links or copy files
            target_dir = os.path.join(swift_resources_dir, os.path.basename(model_path))
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            os.symlink(os.path.abspath(model_path), target_dir)

        logger.info(f"Resources bundled at {swift_resources_dir}")

    # Summary
    if not converted_models:
        logger.warning(
            "No components were converted. Please specify at least one component to convert."
        )
    else:
        logger.info(
            f"Successfully converted {len(converted_models)} components: {', '.join(converted_models.keys())}"
        )


if __name__ == "__main__":
    # Example usage:
    # python -m coreml.torch2coreml --convert-voltron --convert-language-goal --convert-visual-goal --convert-gcdenoiser --output-dir mdt_mlpackage train_folder=pretrained_models/mdt/CALVIN\ D/mdtv_1_d
    main()
