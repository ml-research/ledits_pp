# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import os
# from itertools import repeat
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
    AttnProcessor,
    Attention
)
from diffusers.schedulers import DDIMScheduler
from diffusers.utils import (
    is_accelerate_available,
    is_accelerate_version,
    is_invisible_watermark_available,
    logging,
    randn_tensor,
    replace_example_docstring,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from dataclasses import dataclass
from diffusers.utils import BaseOutput
from .scheduling_dpmsolver_multistep_inject import DPMSolverMultistepSchedulerInject
import math
import torch.nn.functional as F

@dataclass
class StableDiffusionXLPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[Image.Image], np.ndarray]


if is_invisible_watermark_available():
    from .watermark import StableDiffusionXLWatermarker

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionXLPipeline

        >>> pipe = StableDiffusionXLPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""

class AttentionStore():
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def __call__(self, attn, is_cross: bool, place_in_unet: str, editing_prompts):
        # attn.shape = batch_size * head_size, seq_len query, seq_len_key
        if attn.shape[1] <= self.max_size:
            bs = 2 + editing_prompts
            skip = 1  # skip unconditional

            head_size = int(attn.shape[0] / self.batch_size)
            attn = torch.stack(attn.split(self.batch_size)).permute(1, 0, 2, 3)
            source_batch_size = int(attn.shape[1] // bs)
            self.forward(
                attn[:, skip * source_batch_size:],
                is_cross,
                place_in_unet)

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        self.step_store[key].append(attn)

    def between_steps(self, store_step=True):
        if store_step:
            if self.average:
                if len(self.attention_store) == 0:
                    self.attention_store = self.step_store
                else:
                    for key in self.attention_store:
                        for i in range(len(self.attention_store[key])):
                            self.attention_store[key][i] += self.step_store[key][i]
            else:
                if len(self.attention_store) == 0:
                    self.attention_store = [self.step_store]
                else:
                    self.attention_store.append(self.step_store)

            self.cur_step += 1
        self.step_store = self.get_empty_store()

    def get_attention(self, step: int):
        if self.average:
            attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                         self.attention_store}
        else:
            assert (step is not None)
            attention = self.attention_store[step]
        return attention

    def aggregate_attention(self, attention_maps, prompts, res: int,
                            from_where: List[str], is_cross: bool, select: int
                            ):
        out = [[] for x in range(self.batch_size)]
        num_pixels = res ** 2
        for location in from_where:
            for bs_item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                for batch, item in enumerate(bs_item):
                    if item.shape[1] == num_pixels:
                        cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                        out[batch].append(cross_maps)

        out = torch.stack([torch.cat(x, dim=0) for x in out])
        # average over heads
        out = out.sum(1) / out.shape[1]
        return out

    def __init__(self, average: bool, batch_size=1, max_resolution=32):
        self.step_store = self.get_empty_store()
        self.attention_store = []
        self.cur_step = 0
        self.average = average
        self.batch_size = batch_size
        self.max_size = max_resolution ** 2


class CrossAttnProcessor:

    def __init__(self, attention_store, place_in_unet, editing_prompts):
        self.attnstore = attention_store
        self.place_in_unet = place_in_unet
        self.editing_prompts = editing_prompts

    def __call__(
            self,
            attn: Attention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
    ):
        assert (not attn.residual_connection)
        assert (attn.spatial_norm is None)
        assert (attn.group_norm is None)
        assert (hidden_states.ndim != 4)
        assert (encoder_hidden_states is not None)  # is cross

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        self.attnstore(attention_probs,
                       is_cross=True,
                       place_in_unet=self.place_in_unet,
                       editing_prompts=self.editing_prompts)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# Modified from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionAttendAndExcitePipeline.GaussianSmoothing
class GaussianSmoothing():

    def __init__(self, device):
        kernel_size = [3, 3]
        sigma = [0.5, 0.5]

        # The gaussian kernel is the product of the gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(1, *[1] * (kernel.dim() - 1))

        self.weight = kernel.to(device)

    def __call__(self, input):
        """
        Arguments:
        Apply gaussian filter to input.
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return F.conv2d(input, weight=self.weight.to(input.dtype))

def reset_dpm(scheduler):
    if isinstance(scheduler, DPMSolverMultistepSchedulerInject):
        scheduler.model_outputs = [
                                      None,
                                  ] * scheduler.config.solver_order
        scheduler.lower_order_nums = 0

def load_image(image_path, size, left=0, right=0, top=0, bottom=0, device=None, dtype=None):
    print(f"load image of size {size}x{size}")

    def pre_process(im, size, left=0, right=0, top=0, bottom=0):
        if type(im) is str:
            image = np.array(Image.open(im).convert('RGB'))[:, :, :3]
        elif isinstance(im, Image.Image):
            image = np.array((im).convert('RGB'))[:, :, :3]
        else:
            image = im
        h, w, c = image.shape
        left = min(left, w - 1)
        right = min(right, w - left - 1)
        top = min(top, h - left - 1)
        bottom = min(bottom, h - top - 1)
        image = image[top:h - bottom, left:w - right]
        h, w, c = image.shape
        if h < w:
            offset = (w - h) // 2
            image = image[:, offset:offset + h]
        elif w < h:
            offset = (h - w) // 2
            image = image[offset:offset + w]
        image = np.array(Image.fromarray(image).resize((size, size)))
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        return image

    tmps = []
    if isinstance(image_path, list):
        for item in image_path:
            tmps.append(pre_process(item, size, left, right, top, bottom))
    else:
        tmps.append(pre_process(image_path, size, left, right, top, bottom))
    image = torch.stack(tmps) / 127.5 - 1

    image = image.to(device=device, dtype=dtype)
    return image

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class StableDiffusionPipelineXL_LEDITS(DiffusionPipeline, FromSingleFileMixin, LoraLoaderMixin):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion XL.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    In addition the pipeline inherits the following loading methods:
        - *LoRA*: [`StableDiffusionXLPipeline.load_lora_weights`]
        - *Ckpt*: [`loaders.FromSingleFileMixin.from_single_file`]

    as well as the following saving methods:
        - *LoRA*: [`loaders.StableDiffusionXLPipeline.save_lora_weights`]

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion XL uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([` CLIPTextModelWithProjection`]):
            Second frozen text-encoder. Stable Diffusion XL uses the text and pool portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """

    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            text_encoder_2: CLIPTextModelWithProjection,
            tokenizer: CLIPTokenizer,
            tokenizer_2: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler: DDIMScheduler,
            force_zeros_for_empty_prompt: bool = True,
            add_watermarker: Optional[bool] = None,
    ):
        super().__init__()

        if not isinstance(scheduler, DDIMScheduler) or not isinstance(scheduler, DPMSolverMultistepSchedulerInject):
            scheduler = DPMSolverMultistepSchedulerInject.from_config(scheduler.config, algorithm_type="sde-dpmsolver++", solver_order=2)
            logger.warning("This pipeline only supports DDIMScheduler and DPMSolverMultistepSchedulerInject. "
                           "The scheduler has been changed to DPMSolverMultistepSchedulerInject.")

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
        )
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.default_sample_size = self.unet.config.sample_size

        add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

        if add_watermarker:
            self.watermark = StableDiffusionXLWatermarker()
        else:
            self.watermark = None

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_tiling
    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_tiling
    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        model_sequence = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )
        model_sequence.extend([self.unet, self.vae])

        hook = None
        for cpu_offloaded_model in model_sequence:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    def encode_prompt(
            self,
            prompt: str,
            prompt_2: Optional[str] = None,
            device: Optional[torch.device] = None,
            num_images_per_prompt: int = 1,
            do_classifier_free_guidance: bool = True,
            negative_prompt: Optional[str] = None,
            negative_prompt_2: Optional[str] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            lora_scale: Optional[float] = None,
            enable_edit_guidance: bool = True,
            editing_prompt: Optional[str] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )
        num_edit_tokens = 0
        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            # textual inversion: procecss multi-vector tokens if necessary
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]
            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, tokenizer)

                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                        text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1: -1])
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )

                prompt_embeds = text_encoder(
                    text_input_ids.to(device),
                    output_hidden_states=True,
                )

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt, negative_prompt_2]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]

            negative_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    negative_prompt = self.maybe_convert_prompt(negative_prompt, tokenizer)

                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device),
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        if enable_edit_guidance:
            editing_prompt_2 = editing_prompt

            editing_prompts = [editing_prompt, editing_prompt_2]
            edit_prompt_embeds_list = []

            for editing_prompt, tokenizer, text_encoder in zip(editing_prompts, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    editing_prompt = self.maybe_convert_prompt(editing_prompt, tokenizer)

                max_length = prompt_embeds.shape[1]
                edit_concepts_input = tokenizer(
                    # [x for item in editing_prompt for x in repeat(item, batch_size)],
                    editing_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                    return_length=True
                )
                num_edit_tokens = edit_concepts_input.length - 2

                edit_concepts_input_ids = edit_concepts_input.input_ids
                edit_concepts_embeds = text_encoder(
                    edit_concepts_input.input_ids.to(device),
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                edit_pooled_prompt_embeds = edit_concepts_embeds[0]
                edit_concepts_embeds = edit_concepts_embeds.hidden_states[-2]

                edit_prompt_embeds_list.append(edit_concepts_embeds)

            edit_concepts_embeds = torch.concat(edit_prompt_embeds_list, dim=-1)
        else:
            edit_concepts_embeds = None
            edit_pooled_prompt_embeds = None

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if enable_edit_guidance:
            bs_embed_edit, seq_len, _ = edit_concepts_embeds.shape
            edit_concepts_embeds = edit_concepts_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            edit_concepts_embeds = edit_concepts_embeds.repeat(1, num_images_per_prompt, 1)
            edit_concepts_embeds = edit_concepts_embeds.view(bs_embed_edit * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )

        if enable_edit_guidance:
            edit_pooled_prompt_embeds = edit_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed_edit * num_images_per_prompt, -1
            )

        return (prompt_embeds, negative_prompt_embeds, edit_concepts_embeds,
                pooled_prompt_embeds, negative_pooled_prompt_embeds, edit_pooled_prompt_embeds, num_edit_tokens)

    # Modified from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        return extra_step_kwargs

    def check_inputs(
            self,
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt=None,
            negative_prompt_2=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
                callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

    # Modified from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, latents):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)

        if latents.shape != shape:
            raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")

        latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
                self.unet.config.addition_time_embed_dim * len(add_time_ids) + self.text_encoder_2.config.projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae
    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    def prepare_unet(self, attention_store):
        attn_procs = {}
        for name in self.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            if "attn2" in name and place_in_unet not in 'mid':
                attn_procs[name] = CrossAttnProcessor(
                    attention_store=attention_store,
                    place_in_unet=place_in_unet,
                    editing_prompts=self.enabled_editing_prompts)
            else:
                attn_procs[name] = AttnProcessor()

        self.unet.set_attn_processor(attn_procs)

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
            self,
            prompt: Union[str, List[str]] = '',
            prompt_2: Optional[Union[str, List[str]]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            # num_inference_steps: int = 50,
            # denoising_end: Optional[float] = None,
            guidance_scale: float = 5.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            # num_images_per_prompt: Optional[int] = 1,
            # eta: float = 0.0,
            # generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            original_size: Optional[Tuple[int, int]] = None,
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            target_size: Optional[Tuple[int, int]] = None,
            editing_prompt: Optional[Union[str, List[str]]] = None,
            editing_prompt_embeddings: Optional[torch.Tensor] = None,
            reverse_editing_direction: Optional[Union[bool, List[bool]]] = False,
            edit_guidance_scale: Optional[Union[float, List[float]]] = 5,
            edit_warmup_steps: Optional[Union[int, List[int]]] = 10,
            edit_cooldown_steps: Optional[Union[int, List[int]]] = None,
            edit_threshold: Optional[Union[float, List[float]]] = 0.9,
            edit_momentum_scale: Optional[float] = 0.1,
            edit_mom_beta: Optional[float] = 0.4,
            edit_weights: Optional[List[float]] = None,
            sem_guidance: Optional[List[torch.Tensor]] = None,
            use_cross_attn_mask: bool = False,
            # Attention store (just for visualization purposes)
            attn_store_steps: Optional[List[int]] = [],
            store_averaged_over_steps: bool = True,
            use_intersect_mask: bool = False,
            verbose=True
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            denoising_end (`float`, *optional*):
                When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
                completed before it is intentionally prematurely terminated. As a result, the returned sample will
                still retain a substantial amount of noise as determined by the discrete timesteps selected by the
                scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a
                "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
                Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                `original_size` defaults to `(width, height)` if not specified. Part of SDXL's micro-conditioning as
                explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(width, height)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            editing_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to use for semantic guidance. Semantic guidance is disabled by setting
                `editing_prompt = None`. Guidance direction of prompt should be specified via
                `reverse_editing_direction`.
            editing_prompt_embeddings (`torch.Tensor`, *optional*):
                Pre-computed embeddings to use for semantic guidance. Guidance direction of embedding should be
                specified via `reverse_editing_direction`.
            reverse_editing_direction (`bool` or `List[bool]`, *optional*, defaults to `False`):
                Whether the corresponding prompt in `editing_prompt` should be increased or decreased.
            edit_guidance_scale (`float` or `List[float]`, *optional*, defaults to 5):
                Guidance scale for semantic guidance. If provided as a list, values should correspond to
                `editing_prompt`.
            edit_warmup_steps (`float` or `List[float]`, *optional*, defaults to 10):
                Number of diffusion steps (for each prompt) for which semantic guidance is not applied. Momentum is
                calculated for those steps and applied once all warmup periods are over.
            edit_cooldown_steps (`float` or `List[float]`, *optional*, defaults to `None`):
                Number of diffusion steps (for each prompt) after which semantic guidance is longer applied.
            edit_threshold (`float` or `List[float]`, *optional*, defaults to 0.9):
                Threshold of semantic guidance.
            edit_momentum_scale (`float`, *optional*, defaults to 0.1):
                Scale of the momentum to be added to the semantic guidance at each diffusion step. If set to 0.0,
                momentum is disabled. Momentum is already built up during warmup (for diffusion steps smaller than
                `sld_warmup_steps`). Momentum is only added to latent guidance once all warmup periods are finished.
            edit_mom_beta (`float`, *optional*, defaults to 0.4):
                Defines how semantic guidance momentum builds up. `edit_mom_beta` indicates how much of the previous
                momentum is kept. Momentum is already built up during warmup (for diffusion steps smaller than
                `edit_warmup_steps`).
            edit_weights (`List[float]`, *optional*, defaults to `None`):
                Indicates how much each individual concept should influence the overall guidance. If no weights are
                provided all concepts are applied equally.
            sem_guidance (`List[torch.Tensor]`, *optional*):
                List of pre-generated guidance vectors to be applied at generation. Length of the list has to
                correspond to `num_inference_steps`.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """
        eta = self.eta
        num_inference_steps = self.num_inversion_steps
        num_images_per_prompt = 1
        latents = self.init_latents

        use_ddpm = True
        zs = self.zs
        reset_dpm(self.scheduler)

        if use_intersect_mask:
            use_cross_attn_mask = True

        if use_cross_attn_mask:
            self.smoothing = GaussianSmoothing(self.device)

        # 0. Default height and width to unet
        height = self.height
        width = self.width
        original_size = self.original_size
        target_size = self.target_size

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

        org_prompt = prompt
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        if editing_prompt:
            enable_edit_guidance = True
            if isinstance(editing_prompt, str):
                editing_prompt = [editing_prompt]
            self.enabled_editing_prompts = len(editing_prompt)
        elif editing_prompt_embeddings is not None:
            enable_edit_guidance = True
            self.enabled_editing_prompts = editing_prompt_embeddings.shape[0]
        else:
            self.enabled_editing_prompts = 0
            enable_edit_guidance = False

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if prompt == "" and (prompt_2 == "" or prompt_2 is None):
            # only use uncond noise pred
            guidance_scale = 0.0
            do_classifier_free_guidance = True
        else:
            do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            edit_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            pooled_edit_embeds,
            num_edit_tokens
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            enable_edit_guidance=enable_edit_guidance,
            editing_prompt=editing_prompt
        )

        # 4. Prepare timesteps
        # self.scheduler.set_timesteps(num_inference_steps, device=device)

        timesteps = self.scheduler.timesteps
        if use_ddpm:
            t_to_idx = {int(v): k for k, v in enumerate(timesteps)}

        if use_cross_attn_mask:
            self.attention_store = AttentionStore(average=store_averaged_over_steps, batch_size=batch_size)
            self.prepare_unet(self.attention_store)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            latents,
        )

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )

        self.text_cross_attention_maps = [org_prompt] if isinstance(org_prompt, str) else org_prompt

        if enable_edit_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds, edit_prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds, pooled_edit_embeds], dim=0)
            edit_concepts_time_ids = add_time_ids.repeat(edit_prompt_embeds.shape[0], 1)
            add_time_ids = torch.cat([add_time_ids, add_time_ids, edit_concepts_time_ids], dim=0)
            self.text_cross_attention_maps += \
                ([editing_prompt] if isinstance(editing_prompt, str) else editing_prompt)
        elif do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 8. Denoising loop
        self.uncond_estimates = None
        self.text_estimates = None
        self.edit_estimates = None
        self.sem_guidance = None
        self.activation_mask = None

        for i, t in enumerate(self.progress_bar(timesteps, verbose=verbose)):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * (2 + self.enabled_editing_prompts)) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_out = noise_pred.chunk(2 + self.enabled_editing_prompts)  # [b,4, 64, 64]
                    noise_pred_uncond, noise_pred_text = noise_pred_out[0], noise_pred_out[1]
                    noise_pred_edit_concepts = noise_pred_out[2:]

                    # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    noise_guidance = guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if self.uncond_estimates is None:
                        self.uncond_estimates = torch.zeros((len(timesteps), *noise_pred_uncond.shape))
                    self.uncond_estimates[i] = noise_pred_uncond.detach().cpu()

                    if self.text_estimates is None:
                        self.text_estimates = torch.zeros((len(timesteps), *noise_pred_text.shape))
                    self.text_estimates[i] = noise_pred_text.detach().cpu()

                    if self.edit_estimates is None and enable_edit_guidance:
                        self.edit_estimates = torch.zeros(
                            (len(timesteps), len(noise_pred_edit_concepts), *noise_pred_edit_concepts[0].shape)
                        )

                    if self.sem_guidance is None:
                        self.sem_guidance = torch.zeros((len(timesteps), *noise_pred_text.shape))

                    if self.activation_mask is None:
                        self.activation_mask = torch.zeros(
                            (len(timesteps), len(noise_pred_edit_concepts), *noise_pred_edit_concepts[0].shape)
                        )

                    if enable_edit_guidance:
                        concept_weights = torch.zeros(
                            (len(noise_pred_edit_concepts), noise_guidance.shape[0]),
                            device=self.device,
                            dtype=noise_guidance.dtype,
                        )
                        noise_guidance_edit = torch.zeros(
                            (len(noise_pred_edit_concepts), *noise_guidance.shape),
                            device=self.device,
                            dtype=noise_guidance.dtype,
                        )
                        # noise_guidance_edit = torch.zeros_like(noise_guidance)
                        warmup_inds = []
                        for c, noise_pred_edit_concept in enumerate(noise_pred_edit_concepts):
                            self.edit_estimates[i, c] = noise_pred_edit_concept
                            if isinstance(edit_guidance_scale, list):
                                edit_guidance_scale_c = edit_guidance_scale[c]
                            else:
                                edit_guidance_scale_c = edit_guidance_scale

                            if isinstance(edit_threshold, list):
                                edit_threshold_c = edit_threshold[c]
                            else:
                                edit_threshold_c = edit_threshold
                            if isinstance(reverse_editing_direction, list):
                                reverse_editing_direction_c = reverse_editing_direction[c]
                            else:
                                reverse_editing_direction_c = reverse_editing_direction
                            if edit_weights:
                                edit_weight_c = edit_weights[c]
                            else:
                                edit_weight_c = 1.0
                            if isinstance(edit_warmup_steps, list):
                                edit_warmup_steps_c = edit_warmup_steps[c]
                            else:
                                edit_warmup_steps_c = edit_warmup_steps

                            if isinstance(edit_cooldown_steps, list):
                                edit_cooldown_steps_c = edit_cooldown_steps[c]
                            elif edit_cooldown_steps is None:
                                edit_cooldown_steps_c = i + 1
                            else:
                                edit_cooldown_steps_c = edit_cooldown_steps
                            if i >= edit_warmup_steps_c:
                                warmup_inds.append(c)
                            if i >= edit_cooldown_steps_c:
                                noise_guidance_edit[c, :, :, :, :] = torch.zeros_like(noise_pred_edit_concept)
                                continue

                            noise_guidance_edit_tmp = noise_pred_edit_concept - noise_pred_uncond
                            # tmp_weights = (noise_pred_text - noise_pred_edit_concept).sum(dim=(1, 2, 3))
                            tmp_weights = (noise_guidance - noise_pred_edit_concept).sum(dim=(1, 2, 3))

                            tmp_weights = torch.full_like(tmp_weights, edit_weight_c)  # * (1 / enabled_editing_prompts)
                            if reverse_editing_direction_c:
                                noise_guidance_edit_tmp = noise_guidance_edit_tmp * -1
                            concept_weights[c, :] = tmp_weights

                            noise_guidance_edit_tmp = noise_guidance_edit_tmp * edit_guidance_scale_c

                            if use_cross_attn_mask:
                                out = self.attention_store.aggregate_attention(
                                    attention_maps=self.attention_store.step_store,
                                    prompts=self.text_cross_attention_maps,
                                    res=32,
                                    from_where=["up", "down"],
                                    is_cross=True,
                                    select=self.text_cross_attention_maps.index(editing_prompt[c]),
                                )
                                attn_map = out[:, :, :, 1:1 + num_edit_tokens[c]]  # 0 -> startoftext

                                # average over all tokens
                                assert (attn_map.shape[3] == num_edit_tokens[c])
                                attn_map = torch.sum(attn_map, dim=3)

                                # gaussian_smoothing
                                attn_map = F.pad(attn_map.unsqueeze(1), (1, 1, 1, 1), mode="reflect")
                                attn_map = self.smoothing(attn_map).squeeze(1)

                                # create binary mask
                                tmp = torch.quantile(attn_map.flatten(start_dim=1), edit_threshold_c, dim=1)
                                attn_mask = torch.where(attn_map >= tmp.unsqueeze(1).unsqueeze(1).repeat(1, 32, 32),
                                                        1.0,
                                                        0.0)

                                # resolution must match latent space dimension
                                attn_mask = F.interpolate(
                                    attn_mask.unsqueeze(1),
                                    noise_guidance_edit_tmp.shape[-2:]  # 64,64
                                ).repeat(1, 4, 1, 1)
                                self.activation_mask[i, c] = attn_mask.detach().cpu()
                                if not use_intersect_mask:
                                    noise_guidance_edit_tmp = noise_guidance_edit_tmp * attn_mask

                            if use_intersect_mask:
                                noise_guidance_edit_tmp_quantile = torch.abs(noise_guidance_edit_tmp)
                                noise_guidance_edit_tmp_quantile = torch.sum(noise_guidance_edit_tmp_quantile, dim=1,
                                                                            keepdim=True)
                                noise_guidance_edit_tmp_quantile = noise_guidance_edit_tmp_quantile.repeat(1, 4, 1, 1)

                                # torch.quantile function expects float32
                                if noise_guidance_edit_tmp_quantile.dtype == torch.float32:
                                    tmp = torch.quantile(
                                        noise_guidance_edit_tmp_quantile.flatten(start_dim=2),
                                        edit_threshold_c,
                                        dim=2,
                                        keepdim=False,
                                    )
                                else:
                                    tmp = torch.quantile(
                                        noise_guidance_edit_tmp_quantile.flatten(start_dim=2).to(torch.float32),
                                        edit_threshold_c,
                                        dim=2,
                                        keepdim=False,
                                    ).to(noise_guidance_edit_tmp_quantile.dtype)

                                intersect_mask = torch.where(
                                    noise_guidance_edit_tmp_quantile >= tmp[:, :, None, None],
                                    torch.ones_like(noise_guidance_edit_tmp),
                                    torch.zeros_like(noise_guidance_edit_tmp),
                                ) * attn_mask

                                self.activation_mask[i, c] = intersect_mask.detach().cpu()

                                noise_guidance_edit_tmp = noise_guidance_edit_tmp * intersect_mask

                            elif not use_cross_attn_mask:
                                # calculate quantile
                                noise_guidance_edit_tmp_quantile = torch.abs(noise_guidance_edit_tmp)
                                noise_guidance_edit_tmp_quantile = torch.sum(noise_guidance_edit_tmp_quantile, dim=1,
                                                                             keepdim=True)
                                noise_guidance_edit_tmp_quantile = noise_guidance_edit_tmp_quantile.repeat(1, 4, 1, 1)

                                # torch.quantile function expects float32
                                if noise_guidance_edit_tmp_quantile.dtype == torch.float32:
                                    tmp = torch.quantile(
                                        noise_guidance_edit_tmp_quantile.flatten(start_dim=2),
                                        edit_threshold_c,
                                        dim=2,
                                        keepdim=False,
                                    )
                                else:
                                    tmp = torch.quantile(
                                        noise_guidance_edit_tmp_quantile.flatten(start_dim=2).to(torch.float32),
                                        edit_threshold_c,
                                        dim=2,
                                        keepdim=False,
                                    ).to(noise_guidance_edit_tmp_quantile.dtype)

                                self.activation_mask[i, c] = torch.where(
                                    noise_guidance_edit_tmp_quantile >= tmp[:, :, None, None],
                                    torch.ones_like(noise_guidance_edit_tmp),
                                    torch.zeros_like(noise_guidance_edit_tmp),
                                ).detach().cpu()

                                noise_guidance_edit_tmp = torch.where(
                                    noise_guidance_edit_tmp_quantile >= tmp[:, :, None, None],
                                    noise_guidance_edit_tmp,
                                    torch.zeros_like(noise_guidance_edit_tmp),
                                )

                            noise_guidance_edit[c, :, :, :, :] = noise_guidance_edit_tmp

                        warmup_inds = torch.tensor(warmup_inds).to(self.device)

                        if warmup_inds.shape[0] == len(noise_pred_edit_concepts):
                            warmup_inds = torch.tensor(warmup_inds).to(self.device)

                            concept_weights = torch.where(
                                concept_weights < 0, torch.zeros_like(concept_weights), concept_weights
                            )

                            concept_weights = torch.nan_to_num(concept_weights)

                            noise_guidance_edit = torch.einsum("cb,cbijk->bijk", concept_weights, noise_guidance_edit)
                            noise_guidance = noise_guidance + noise_guidance_edit
                            self.sem_guidance[i] = noise_guidance_edit.detach().cpu()

                    if sem_guidance is not None:
                        edit_guidance = sem_guidance[i].to(self.device)
                        noise_guidance = noise_guidance + edit_guidance

                    noise_pred = noise_pred_uncond + noise_guidance

                # TODO: compatible with SEGA?
                # if do_classifier_free_guidance and guidance_rescale > 0.0:
                #    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                #    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                if use_ddpm:
                    idx = t_to_idx[int(t)]
                    latents = self.scheduler.step(noise_pred, t, latents, variance_noise=zs[idx],
                                                  **extra_step_kwargs).prev_sample

                else:  # if not use_ddpm:
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                if use_cross_attn_mask:
                    # step callback
                    store_step = i in attn_store_steps
                    if store_step:
                        print(f"storing attention for step {i}")
                    self.attention_store.between_steps(store_step)

                # call the callback, if provided
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

        # make sure the VAE is in float32 mode, as it overflows in float16
        if self.vae.dtype == torch.float16 and self.vae.config.force_upcast:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
        elif self.vae.config.force_upcast:
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents
            return StableDiffusionXLPipelineOutput(images=image)

        # apply watermark if available
        if self.watermark is not None:
            image = self.watermark.apply_watermark(image)

        image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)

    # Overrride to properly handle the loading and unloading of the additional text encoder.
    def load_lora_weights(self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]], **kwargs):
        # We could have accessed the unet config from `lora_state_dict()` too. We pass
        # it here explicitly to be able to tell that it's coming from an SDXL
        # pipeline.
        state_dict, network_alphas = self.lora_state_dict(
            pretrained_model_name_or_path_or_dict,
            unet_config=self.unet.config,
            **kwargs,
        )
        self.load_lora_into_unet(state_dict, network_alphas=network_alphas, unet=self.unet)

        text_encoder_state_dict = {k: v for k, v in state_dict.items() if "text_encoder." in k}
        if len(text_encoder_state_dict) > 0:
            self.load_lora_into_text_encoder(
                text_encoder_state_dict,
                network_alphas=network_alphas,
                text_encoder=self.text_encoder,
                prefix="text_encoder",
                lora_scale=self.lora_scale,
            )

        text_encoder_2_state_dict = {k: v for k, v in state_dict.items() if "text_encoder_2." in k}
        if len(text_encoder_2_state_dict) > 0:
            self.load_lora_into_text_encoder(
                text_encoder_2_state_dict,
                network_alphas=network_alphas,
                text_encoder=self.text_encoder_2,
                prefix="text_encoder_2",
                lora_scale=self.lora_scale,
            )

    @classmethod
    def save_lora_weights(
            self,
            save_directory: Union[str, os.PathLike],
            unet_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
            text_encoder_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
            text_encoder_2_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
            is_main_process: bool = True,
            weight_name: str = None,
            save_function: Callable = None,
            safe_serialization: bool = True,
    ):
        state_dict = {}

        def pack_weights(layers, prefix):
            layers_weights = layers.state_dict() if isinstance(layers, torch.nn.Module) else layers
            layers_state_dict = {f"{prefix}.{module_name}": param for module_name, param in layers_weights.items()}
            return layers_state_dict

        state_dict.update(pack_weights(unet_lora_layers, "unet"))

        if text_encoder_lora_layers and text_encoder_2_lora_layers:
            state_dict.update(pack_weights(text_encoder_lora_layers, "text_encoder"))
            state_dict.update(pack_weights(text_encoder_2_lora_layers, "text_encoder_2"))

        self.write_lora_layers(
            state_dict=state_dict,
            save_directory=save_directory,
            is_main_process=is_main_process,
            weight_name=weight_name,
            save_function=save_function,
            safe_serialization=safe_serialization,
        )

    def _remove_text_encoder_monkey_patch(self):
        self._remove_text_encoder_monkey_patch_classmethod(self.text_encoder)
        self._remove_text_encoder_monkey_patch_classmethod(self.text_encoder_2)

    def progress_bar(self, iterable=None, total=None, verbose=True):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )
        if not verbose:
            return iterable
        elif iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    @torch.no_grad()
    def invert(self,
               image_path: str,
               source_prompt: str = "",
               source_prompt_2: str = None,
               source_guidance_scale=3.5,
               negative_prompt: str = None,
               negative_prompt_2: str = None,
               num_inversion_steps: int = 100,
               skip: float = .15,
               eta: float = 1.0,
               generator: Optional[torch.Generator] = None,
               height: Optional[int] = None,
               width: Optional[int] = None,
               original_size: Optional[Tuple[int, int]] = None,
               crops_coords_top_left: Tuple[int, int] = (0, 0),
               target_size: Optional[Tuple[int, int]] = None,
               num_zero_noise_steps=0,
               verbose=True
               ):
        """
        Inverts a real image according to Algorihm 1 in https://arxiv.org/pdf/2304.06140.pdf,
        based on the code in https://github.com/inbarhub/DDPM_inversion

        returns:
        zs - noise maps
        xts - intermediate inverted latents
        """

        self.eta = eta
        assert (self.eta > 0)

        train_steps = self.scheduler.config.num_train_timesteps
        timesteps = torch.from_numpy(
            np.linspace(train_steps - skip * train_steps - 1, 1, num_inversion_steps).astype(np.int64)).to(self.device)

        self.num_inversion_steps = timesteps.shape[0]
        self.scheduler.num_inference_steps = timesteps.shape[0]
        self.scheduler.timesteps = timesteps
        reset_dpm(self.scheduler)

        cross_attention_kwargs = None  # TODO
        batch_size = 1

        num_images_per_prompt = 1

        device = self._execution_device

        # Reset attn processor, we do not want to store attn maps during inversion
        self.unet.set_default_attn_processor()

        # 0. Ensure that only uncond embedding is used if prompt = ""
        if source_prompt == "" and \
                (source_prompt_2 == "" or source_prompt_2 is None):
            # noise pred should only be noise_pred_uncond
            source_guidance_scale = 0.0
            do_classifier_free_guidance = True
        else:
            do_classifier_free_guidance = source_guidance_scale > 1.0

        # 1. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        self.height = height
        self.width = width
        self.original_size = original_size
        self.target_size = target_size

        # 2. get embeddings
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            _,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            _,
            _
        ) = self.encode_prompt(
            prompt=source_prompt,
            prompt_2=source_prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            lora_scale=text_encoder_lora_scale,
            enable_edit_guidance=False,
        )

        # 3. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 4. prepare image
        image = Image.open(image_path)
        size = self.unet.sample_size * self.vae_scale_factor
        image = image.convert("RGB").resize((size, size))
        image = self.image_processor.preprocess(image)
        image = image.to(device=device, dtype=negative_prompt_embeds.dtype)

        if image.shape[1] == 4:
            x0 = image
        else:
            if self.vae.config.force_upcast:
                image = image.float()
                self.vae.to(dtype=torch.float32)

            x0 = self.vae.encode(image).latent_dist.sample(generator)
            x0 = x0.to(negative_prompt_embeds.dtype)
            x0 = self.vae.config.scaling_factor * x0

        # autoencoder reconstruction
        if self.vae.dtype == torch.float16 and self.vae.config.force_upcast:
            self.upcast_vae()
            x0_tmp = x0.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            image_rec = self.vae.decode(x0_tmp / self.vae.config.scaling_factor, return_dict=False)[0]
        elif self.vae.config.force_upcast:
            x0_tmp = x0.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            image_rec = self.vae.decode(x0_tmp / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image_rec = self.vae.decode(x0 / self.vae.config.scaling_factor, return_dict=False)[0]

        image_rec = self.image_processor.postprocess(image_rec, output_type="pil")

        # 5. find zs and xts
        variance_noise_shape = (
            self.num_inversion_steps,
            self.unet.config.in_channels,
            self.unet.sample_size,
            self.unet.sample_size)

        # intermediate latents
        t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
        xts = torch.zeros(size=variance_noise_shape, device=self.device, dtype=negative_prompt_embeds.dtype)

        for t in reversed(timesteps):
            idx = self.num_inversion_steps - t_to_idx[int(t)] - 1
            noise = randn_tensor(shape=x0.shape, generator=generator, device=self.device, dtype=x0.dtype)
            xts[idx] = self.scheduler.add_noise(x0, noise, t)
        xts = torch.cat([x0, xts], dim=0)

        # noise maps
        zs = torch.zeros(size=variance_noise_shape, device=self.device, dtype=negative_prompt_embeds.dtype)

        for t in self.progress_bar(timesteps, verbose=verbose):
            idx = self.num_inversion_steps - t_to_idx[int(t)] - 1
            # 1. predict noise residual
            xt = xts[idx + 1][None]

            latent_model_input = (
                torch.cat([xt] * 2) if do_classifier_free_guidance else xt
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # 2. perform guidance
            if do_classifier_free_guidance:
                noise_pred_out = noise_pred.chunk(2)
                noise_pred_uncond, noise_pred_text = noise_pred_out[0], noise_pred_out[1]
                noise_pred = noise_pred_uncond + source_guidance_scale * (noise_pred_text - noise_pred_uncond)

            xtm1 = xts[idx][None]
            z, xtm1_corrected = compute_noise(self.scheduler, xtm1, xt, t, noise_pred, eta)
            zs[idx] = z

            # correction to avoid error accumulation
            xts[idx] = xtm1_corrected

        self.init_latents = xts[-1].expand(batch_size, -1, -1, -1)
        zs = zs.flip(0)
        for i in range(num_zero_noise_steps):
            zs[-1-i] = torch.zeros_like(zs[-1])

        self.zs = zs
        return zs, xts, image_rec


def compute_noise_ddim(scheduler, prev_latents, latents, timestep, noise_pred, eta):
    # 1. get previous step value (=t-1)
    prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps

    # 2. compute alphas, betas
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
    )

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

    # 4. Clip "predicted x_0"
    if scheduler.config.clip_sample:
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = scheduler._get_variance(timestep, prev_timestep)
    std_dev_t = eta * variance ** (0.5)

    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** (0.5) * noise_pred

    # modifed so that updated xtm1 is returned as well (to avoid error accumulation)
    mu_xt = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    noise = (prev_latents - mu_xt) / (variance ** (0.5) * eta)

    return noise, mu_xt + (eta * variance ** 0.5) * noise


# Copied from pipelines.StableDiffusion.CycleDiffusionPipeline.compute_noise
def compute_noise_sde_dpm_pp_2nd(scheduler, prev_latents, latents, timestep, noise_pred, eta):
    def first_order_update(model_output, timestep, prev_timestep, sample):
        lambda_t, lambda_s = scheduler.lambda_t[prev_timestep], scheduler.lambda_t[timestep]
        alpha_t, alpha_s = scheduler.alpha_t[prev_timestep], scheduler.alpha_t[timestep]
        sigma_t, sigma_s = scheduler.sigma_t[prev_timestep], scheduler.sigma_t[timestep]
        h = lambda_t - lambda_s

        mu_xt = (
                (sigma_t / sigma_s * torch.exp(-h)) * sample
                + (alpha_t * (1 - torch.exp(-2.0 * h))) * model_output
        )
        sigma = sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h))

        noise = (prev_latents - mu_xt) / sigma

        prev_sample = mu_xt + sigma * noise

        return noise, prev_sample

    def second_order_update(model_output_list, timestep_list, prev_timestep, sample):
        t, s0, s1 = prev_timestep, timestep_list[-1], timestep_list[-2]
        m0, m1 = model_output_list[-1], model_output_list[-2]
        lambda_t, lambda_s0, lambda_s1 = scheduler.lambda_t[t], scheduler.lambda_t[s0], scheduler.lambda_t[s1]
        alpha_t, alpha_s0 = scheduler.alpha_t[t], scheduler.alpha_t[s0]
        sigma_t, sigma_s0 = scheduler.sigma_t[t], scheduler.sigma_t[s0]
        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)

        mu_xt = (
                (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                + 0.5 * (alpha_t * (1 - torch.exp(-2.0 * h))) * D1
        )
        sigma = sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h))

        noise = (prev_latents - mu_xt) / sigma

        prev_sample = mu_xt + sigma * noise

        return noise, prev_sample

    step_index = (scheduler.timesteps == timestep).nonzero()
    if len(step_index) == 0:
        step_index = len(scheduler.timesteps) - 1
    else:
        step_index = step_index.item()

    prev_timestep = 0 if step_index == len(scheduler.timesteps) - 1 else scheduler.timesteps[step_index + 1]

    model_output = scheduler.convert_model_output(noise_pred, timestep, latents)

    for i in range(scheduler.config.solver_order - 1):
        scheduler.model_outputs[i] = scheduler.model_outputs[i + 1]
    scheduler.model_outputs[-1] = model_output

    if scheduler.lower_order_nums < 1:
        noise, prev_sample = first_order_update(model_output, timestep, prev_timestep, latents)
    else:
        timestep_list = [scheduler.timesteps[step_index - 1], timestep]
        noise, prev_sample = second_order_update(scheduler.model_outputs, timestep_list, prev_timestep, latents)

    if scheduler.lower_order_nums < scheduler.config.solver_order:
        scheduler.lower_order_nums += 1

    return noise, prev_sample


def compute_noise(scheduler, *args):
    if isinstance(scheduler, DDIMScheduler):
        return compute_noise_ddim(scheduler, *args)
    elif isinstance(scheduler,
                    DPMSolverMultistepSchedulerInject) and scheduler.config.algorithm_type == 'sde-dpmsolver++' \
            and scheduler.config.solver_order == 2:
        return compute_noise_sde_dpm_pp_2nd(scheduler, *args)
    else:
        raise NotImplementedError
