import html
import inspect
import re
import urllib.parse as ul
from typing import Any, Callable, Dict, List, Optional, Union
from itertools import repeat

from diffusers.utils import pt_to_pil

import numpy as np
import PIL
from tqdm import tqdm
import torch
import torch.nn.functional as F
import math
from transformers import CLIPImageProcessor, T5EncoderModel, T5Tokenizer

from diffusers.models.attention_processor import AttnProcessor, Attention, AttnAddedKVProcessor
from diffusers.models import UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler
from diffusers.utils import (
    BACKENDS_MAPPING,
    PIL_INTERPOLATION,
    is_accelerate_available,
    is_accelerate_version,
    is_bs4_available,
    is_ftfy_available,
    logging,
    randn_tensor,
    replace_example_docstring,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.deepfloyd_if import IFPipelineOutput
from diffusers.pipelines.deepfloyd_if.safety_checker import IFSafetyChecker
from diffusers.pipelines.deepfloyd_if.watermark import IFWatermarker

from scheduling_dpmsolver_multistep_inject import DPMSolverMultistepSchedulerInject


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_bs4_available():
    from bs4 import BeautifulSoup

if is_ftfy_available():
    import ftfy

def resize(images: PIL.Image.Image, img_size: int) -> PIL.Image.Image:
    w, h = images.size

    coef = w / h

    w, h = img_size, img_size

    if coef >= 1:
        w = int(round(img_size / 8 * coef) * 8)
    else:
        h = int(round(img_size / 8 / coef) * 8)

    images = images.resize((w, h), resample=PIL_INTERPOLATION["bicubic"], reducing_gap=None)

    return images

def reset_dpm(scheduler):
    if isinstance(scheduler, DPMSolverMultistepSchedulerInject):
        scheduler.model_outputs = [
                                      None,
                                  ] * scheduler.config.solver_order
        scheduler.lower_order_nums = 0


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import IFPipeline, IFSuperResolutionPipeline, DiffusionPipeline
        >>> from diffusers.utils import pt_to_pil
        >>> import torch

        >>> pipe = IFPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
        >>> pipe.enable_model_cpu_offload()

        >>> prompt = 'a photo of a kangaroo wearing an orange hoodie and blue sunglasses standing in front of the eiffel tower holding a sign that says "very deep learning"'
        >>> prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)

        >>> image = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, output_type="pt").images

        >>> # save intermediate image
        >>> pil_image = pt_to_pil(image)
        >>> pil_image[0].save("./if_stage_I.png")

        >>> super_res_1_pipe = IFSuperResolutionPipeline.from_pretrained(
        ...     "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
        ... )
        >>> super_res_1_pipe.enable_model_cpu_offload()

        >>> image = super_res_1_pipe(
        ...     image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, output_type="pt"
        ... ).images

        >>> # save intermediate image
        >>> pil_image = pt_to_pil(image)
        >>> pil_image[0].save("./if_stage_I.png")

        >>> safety_modules = {
        ...     "feature_extractor": pipe.feature_extractor,
        ...     "safety_checker": pipe.safety_checker,
        ...     "watermarker": pipe.watermarker,
        ... }
        >>> super_res_2_pipe = DiffusionPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16
        ... )
        >>> super_res_2_pipe.enable_model_cpu_offload()

        >>> image = super_res_2_pipe(
        ...     prompt=prompt,
        ...     image=image,
        ... ).images
        >>> image[0].save("./if_stage_II.png")
        ```
"""


class AttentionStore():
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def __call__(self, attn, is_cross: bool, place_in_unet: str, editing_prompts, PnP):
        # attn.shape = batch_size * head_size, seq_len query, seq_len_key
        bs = 2 + int(PnP) + editing_prompts
        source_batch_size = int(attn.shape[0] // bs)
        skip = 2 if PnP else 1 # skip PnP & unconditional
        self.forward(
                attn[skip*source_batch_size:],
                is_cross,
                place_in_unet)

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] == 16 ** 2 or attn.shape[1] == 8 ** 2:  # avoid memory overhead
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
            attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        else:
            assert(step is not None)
            attention = self.attention_store[step]
        return attention

    def aggregate_attention(self, attention_maps, prompts, res: int,
        from_where: List[str], is_cross: bool, select: int
    ):
        out = []
        num_pixels = res ** 2
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                    out.append(cross_maps)
        out = torch.cat(out, dim=0)
        # average over heads
        out = out.sum(0) / out.shape[0]
        return out

    def __init__(self, average: bool):
        self.step_store = self.get_empty_store()
        self.attention_store = []
        self.cur_step = 0
        self.average = average

class CrossAttnProcessor:

    def __init__(self, attention_store, place_in_unet, PnP, editing_prompts):
        self.attnstore = attention_store
        self.place_in_unet = place_in_unet
        self.editing_prompts = editing_prompts
        self.PnP = PnP

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states
        hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], -1).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        is_cross = True
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            is_cross = False
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.head_to_batch_dim(encoder_hidden_states_key_proj)
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(encoder_hidden_states_value_proj)

        if not attn.only_cross_attention:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=1)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=1)
        else:
            key = encoder_hidden_states_key_proj
            value = encoder_hidden_states_value_proj

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        if is_cross:
            self.attnstore(attention_probs,
                    is_cross=True,
                    place_in_unet=self.place_in_unet,
                    editing_prompts=self.editing_prompts,
                    PnP=self.PnP)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
        hidden_states = hidden_states + residual

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

class IFDiffusion_LEDITS(DiffusionPipeline):
    tokenizer: T5Tokenizer
    text_encoder: T5EncoderModel

    unet: UNet2DConditionModel
    scheduler: DDIMScheduler

    feature_extractor: Optional[CLIPImageProcessor]
    safety_checker: Optional[IFSafetyChecker]

    watermarker: Optional[IFWatermarker]

    bad_punct_regex = re.compile(
        r"[" + "#®•©™&@·º½¾¿¡§~" + "\)" + "\(" + "\]" + "\[" + "\}" + "\{" + "\|" + "\\" + "\/" + "\*" + r"]{1,}"
    )  # noqa

    _optional_components = ["tokenizer", "text_encoder", "safety_checker", "feature_extractor", "watermarker"]

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        unet: UNet2DConditionModel,
        scheduler: DDIMScheduler,
        safety_checker: Optional[IFSafetyChecker],
        feature_extractor: Optional[CLIPImageProcessor],
        watermarker: Optional[IFWatermarker],
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if not isinstance(scheduler, DDIMScheduler) or not isinstance(scheduler, DPMSolverMultistepSchedulerInject):
            conf = scheduler.config
            conf["clip_sample"] = False
            conf["thresholding"] = False
            conf["variance_type"] = "fixed"

            scheduler = DPMSolverMultistepSchedulerInject.from_config(conf, algorithm_type="sde-dpmsolver++", solver_order=2)
            logger.warning("This pipeline only supports DDIMScheduler and DPMSolverMultistepSchedulerInject. "
                           "The scheduler has been changed to DPMSolverMultistepSchedulerInject.")


        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the IF license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            watermarker=watermarker,
        )
        self.register_to_config(requires_safety_checker=requires_safety_checker)

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

        hook = None

        if self.text_encoder is not None:
            _, hook = cpu_offload_with_hook(self.text_encoder, device, prev_module_hook=hook)

            # Accelerate will move the next model to the device _before_ calling the offload hook of the
            # previous model. This will cause both models to be present on the device at the same time.
            # IF uses T5 for its text encoder which is really large. We can manually call the offload
            # hook for the text encoder to ensure it's moved to the cpu before the unet is moved to
            # the GPU.
            self.text_encoder_offload_hook = hook

        _, hook = cpu_offload_with_hook(self.unet, device, prev_module_hook=hook)

        # if the safety checker isn't called, `unet_offload_hook` will have to be called to manually offload the unet
        self.unet_offload_hook = hook

        if self.safety_checker is not None:
            _, hook = cpu_offload_with_hook(self.safety_checker, device, prev_module_hook=hook)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    def remove_all_hooks(self):
        if is_accelerate_available():
            from accelerate.hooks import remove_hook_from_module
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        for model in [self.text_encoder, self.unet, self.safety_checker]:
            if model is not None:
                remove_hook_from_module(model, recurse=True)

        self.unet_offload_hook = None
        self.text_encoder_offload_hook = None
        self.final_offload_hook = None

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt,
        do_classifier_free_guidance=True,
        num_images_per_prompt=1,
        device=None,
        negative_prompt=None,
        editing_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        edit_prompt_embeds: Optional[torch.FloatTensor] = None,
        clean_caption: bool = False,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`, *optional*):
                torch device to place the resulting embeddings on
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            editing_prompt (`str` or `List[str]`, *optional*):
                The prompt used for semantic guidance
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and negative_prompt is not None:
            if type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )

        if device is None:
            device = self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # while T5 can handle much longer input sequences than 77, the text encoder was trained with a max length of 77 for IF
        max_length = 77

        if prompt_embeds is None:
            prompt = self._text_preprocessing(prompt, clean_caption=clean_caption)
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {max_length} tokens: {removed_text}"
                )

            attention_mask = text_inputs.attention_mask.to(device)

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        if self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        elif self.unet is not None:
            dtype = self.unet.dtype
        else:
            dtype = None

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            uncond_tokens = self._text_preprocessing(uncond_tokens, clean_caption=clean_caption)
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            attention_mask = uncond_input.attention_mask.to(device)

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        num_edit_tokens = 0
        if do_classifier_free_guidance and editing_prompt is not None and edit_prompt_embeds is None:
            edit_tokens: List[str]
            if isinstance(editing_prompt, str):
                edit_tokens = [editing_prompt]
            else:
                edit_tokens = editing_prompt
            edit_tokens = [x for item in edit_tokens for x in repeat(item, batch_size)]
            edit_tokens = self._text_preprocessing(edit_tokens, clean_caption=clean_caption)

            max_length = prompt_embeds.shape[1]
            edit_input = self.tokenizer(
                edit_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
                return_length=True
            )
            num_edit_tokens = edit_input.length -1 # not counting endoftext (there is no startoftext)
            #print(f"num edit tokens: {num_edit_tokens}")

            #edit_tokens = [[word.replace("</w>", "") for word in self.tokenizer.tokenize(item)] for item in editing_prompt]    
            #print(f"edit_tokens: {edit_tokens}")

            attention_mask = edit_input.attention_mask.to(device)

            edit_prompt_embeds = self.text_encoder(
                edit_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            edit_prompt_embeds = edit_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if editing_prompt is not None:
                bs_embed_edit, seq_len_edit, _ = edit_prompt_embeds.shape
                edit_prompt_embeds = edit_prompt_embeds.repeat(1, num_images_per_prompt, 1)
                edit_prompt_embeds = edit_prompt_embeds.view(bs_embed_edit * num_images_per_prompt, seq_len_edit, -1)

        else:
            negative_prompt_embeds = None
            edit_prompt_embeds = None

        return prompt_embeds, negative_prompt_embeds, edit_prompt_embeds, num_edit_tokens

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
            image, nsfw_detected, watermark_detected = self.safety_checker(
                images=image,
                clip_input=safety_checker_input.pixel_values.to(dtype=dtype),
            )
        else:
            nsfw_detected = None
            watermark_detected = None

            if hasattr(self, "unet_offload_hook") and self.unet_offload_hook is not None:
                self.unet_offload_hook.offload()

        return image, nsfw_detected, watermark_detected

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
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
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
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # Modified
    def prepare_intermediate_images(self, batch_size, num_channels, height, width, dtype, device, intermediate_images):
        shape = (batch_size, num_channels, height, width)

        if intermediate_images.shape != shape:
            raise ValueError(f"Unexpected image shape, got {intermediate_images.shape}, expected {shape}")

        intermediate_images = intermediate_images.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        intermediate_images = intermediate_images * self.scheduler.init_noise_sigma
        return intermediate_images

    def prepare_unet(self, attention_store, enabled_editing_prompts):
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

            attn_procs[name] = CrossAttnProcessor(
                attention_store=attention_store,
                place_in_unet=place_in_unet,
                PnP=False,
                editing_prompts=enabled_editing_prompts)

        self.unet.set_attn_processor(attn_procs)

    def _text_preprocessing(self, text, clean_caption=False):
        if clean_caption and not is_bs4_available():
            logger.warn(BACKENDS_MAPPING["bs4"][-1].format("Setting `clean_caption=True`"))
            logger.warn("Setting `clean_caption` to False...")
            clean_caption = False

        if clean_caption and not is_ftfy_available():
            logger.warn(BACKENDS_MAPPING["ftfy"][-1].format("Setting `clean_caption=True`"))
            logger.warn("Setting `clean_caption` to False...")
            clean_caption = False

        if not isinstance(text, (tuple, list)):
            text = [text]

        def process(text: str):
            if clean_caption:
                text = self._clean_caption(text)
                text = self._clean_caption(text)
            else:
                text = text.lower().strip()
            return text

        return [process(t) for t in text]

    def _clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub("<person>", "person", caption)
        # urls:
        caption = re.sub(
            r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        caption = re.sub(
            r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features="html.parser").text

        # @<nickname>
        caption = re.sub(r"@[\w\d]+\b", "", caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
        caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
        caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
        caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
        caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
        caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
        caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",  # noqa
            "-",
            caption,
        )

        # кавычки к одному стандарту
        caption = re.sub(r"[`´«»“”¨]", '"', caption)
        caption = re.sub(r"[‘’]", "'", caption)

        # &quot;
        caption = re.sub(r"&quot;?", "", caption)
        # &amp
        caption = re.sub(r"&amp", "", caption)

        # ip adresses:
        caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

        # article ids:
        caption = re.sub(r"\d:\d\d\s+$", "", caption)

        # \n
        caption = re.sub(r"\\n", " ", caption)

        # "#123"
        caption = re.sub(r"#\d{1,3}\b", "", caption)
        # "#12345.."
        caption = re.sub(r"#\d{5,}\b", "", caption)
        # "123456.."
        caption = re.sub(r"\b\d{6,}\b", "", caption)
        # filenames:
        caption = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption)

        #
        caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

        caption = re.sub(self.bad_punct_regex, r" ", caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, " ", caption)

        caption = ftfy.fix_text(caption)
        caption = html.unescape(html.unescape(caption))

        caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
        caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
        caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

        caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
        caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
        caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
        caption = re.sub(r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption)
        caption = re.sub(r"\bpage\s+\d+\b", "", caption)

        caption = re.sub(r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption)  # j2d1a2a...

        caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

        caption = re.sub(r"\b\s+\:\s+", r": ", caption)
        caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
        caption = re.sub(r"\s+", " ", caption)

        caption.strip()

        caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
        caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
        caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
        caption = re.sub(r"^\.\S+$", "", caption)

        return caption.strip()

    def crop(self,image_path, left=0, right=0, top=0, bottom=0, size=64):
        if type(image_path) is str:
            image = np.array(PIL.Image.open(image_path).convert('RGB'))[:, :, :3]
        else:
            image = image_path
        h, w, c = image.shape
        left = min(left, w-1)
        right = min(right, w - left - 1)
        top = min(top, h - left - 1)
        bottom = min(bottom, h - top - 1)
        image = image[top:h-bottom, left:w-right]
        h, w, c = image.shape
        if h < w:
            offset = (w - h) // 2
            image = image[:, offset:offset + h]
        elif w < h:
            offset = (h - w) // 2
            image = image[offset:offset + w]
        image = PIL.Image.fromarray(image).resize((size, size))
        return image

    # Copied from diffusers.pipelines.deepfloyed_if.IFImg2ImgPipeline.preprocess_image
    def preprocess_image(self, image: PIL.Image.Image) -> torch.Tensor:
        if not isinstance(image, list):
            image = [image]

        def numpy_to_pt(images):
            if images.ndim == 3:
                images = images[..., None]

            images = torch.from_numpy(images.transpose(0, 3, 1, 2))
            return images

        if isinstance(image[0], PIL.Image.Image):
            new_image = []

            for image_ in image:
                image_ = image_.convert("RGB")
                image_ = resize(image_, self.unet.sample_size)
                image_ = np.array(image_)
                image_ = image_.astype(np.float32)
                image_ = image_ / 127.5 - 1
                new_image.append(image_)

            image = new_image

            image = np.stack(image, axis=0)  # to np
            image = numpy_to_pt(image)  # to pt

        elif isinstance(image[0], np.ndarray):
            image = np.concatenate(image, axis=0) if image[0].ndim == 4 else np.stack(image, axis=0)
            image = numpy_to_pt(image)

        elif isinstance(image[0], torch.Tensor):
            image = torch.cat(image, axis=0) if image[0].ndim == 4 else torch.stack(image, axis=0)

        return image

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        #num_inference_steps: int = 100,
        #timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        editing_prompt: Optional[Union[str, List[str]]] = None,
        #num_images_per_prompt: Optional[int] = 1,
        #height: Optional[int] = None,
        #width: Optional[int] = None,
        #eta: float = 0.0,
        #generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        edit_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        reverse_editing_direction: Optional[Union[bool, List[bool]]] = False,
        edit_guidance_scale: Optional[Union[float, List[float]]] = 5,
        edit_warmup_steps: Optional[Union[int, List[int]]] = 10,
        edit_cooldown_steps: Optional[Union[int, List[int]]] = None,
        edit_threshold: Optional[Union[float, List[float]]] = 0.9,
        edit_momentum_scale: Optional[float] = 0.1,
        edit_mom_beta: Optional[float] = 0.4,
        edit_weights: Optional[List[float]] = None,
        #cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        use_cross_attn_mask: bool = False,
        use_intersect_mask: bool = False,
        # Attention store (just for visualization purposes)
        attn_store_steps: Optional[List[int]] = [],
        store_averaged_over_steps: bool = True,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process. If not defined, equal spaced `num_inference_steps`
                timesteps are used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            height (`int`, *optional*, defaults to self.unet.config.sample_size):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size):
                The width in pixels of the generated image.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.IFPipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            clean_caption (`bool`, *optional*, defaults to `True`):
                Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to
                be installed. If the dependencies are not installed, the embeddings will be created from the raw
                prompt.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.IFPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.IFPipelineOutput`] if `return_dict` is True, otherwise a `tuple. When
            returning a tuple, the first element is a list with the generated images, and the second element is a list
            of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work" (nsfw)
            or watermarked content, according to the `safety_checker`.
        """
        eta = self.eta
        num_inference_steps = self.num_inversion_steps
        num_images_per_prompt = 1
        cross_attention_kwargs = None
        intermediate_images = self.init_images

        use_ddpm = True
        zs = self.zs

        reset_dpm(self.scheduler)

        if use_intersect_mask:
            use_cross_attn_mask = True

        if use_cross_attn_mask:
            self.smoothing = GaussianSmoothing(self._execution_device)

        # 0. Default height and width
        height = self.unet.config.sample_size
        width = self.unet.config.sample_size

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

        # 2. Define call parameters
        height = height or self.unet.config.sample_size
        width = width or self.unet.config.sample_size

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if editing_prompt:
            enable_edit_guidance = True
            if isinstance(editing_prompt, str):
                editing_prompt = [editing_prompt]
            enabled_editing_prompts = len(editing_prompt)
        elif edit_prompt_embeds is not None:
            enable_edit_guidance = True
            enabled_editing_prompts = int(edit_prompt_embeds.shape[0] / batch_size)
        else:
            enabled_editing_prompts = 0
            enable_edit_guidance = False

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds, edit_prompt_embeds, num_edit_tokens = self.encode_prompt(
            prompt,
            do_classifier_free_guidance,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            negative_prompt=negative_prompt,
            editing_prompt=editing_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            edit_prompt_embeds=edit_prompt_embeds,
            clean_caption=clean_caption,
        )

        self.text_cross_attention_maps = [prompt] if isinstance(prompt, str) else prompt
        if do_classifier_free_guidance:
            if enable_edit_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds, edit_prompt_embeds])
                self.text_cross_attention_maps += \
                    ([editing_prompt] if isinstance(editing_prompt, str) else editing_prompt)
            else:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        timesteps = self.scheduler.timesteps
        if use_ddpm:
            t_to_idx = {int(v): k for k, v in enumerate(timesteps)}

        if use_cross_attn_mask:
            self.attention_store = AttentionStore(average=store_averaged_over_steps)
            self.prepare_unet(self.attention_store, enabled_editing_prompts)

        # 5. Prepare intermediate images
        intermediate_images = self.prepare_intermediate_images(
            batch_size * num_images_per_prompt,
            self.unet.config.in_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            intermediate_images
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(eta)

        # HACK: see comment in `enable_model_cpu_offload`
        if hasattr(self, "text_encoder_offload_hook") and self.text_encoder_offload_hook is not None:
            self.text_encoder_offload_hook.offload()

        # Initialize edit_momentum to None
        edit_momentum = None
        self.sem_guidance = None

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                model_input = (
                    torch.cat([intermediate_images] * (2 + enabled_editing_prompts)) if do_classifier_free_guidance else intermediate_images
                )
                model_input = self.scheduler.scale_model_input(model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_out = noise_pred.chunk(2 + enabled_editing_prompts)  # [b,4, 64, 64]
                    noise_pred_uncond, noise_pred_text = noise_pred_out[0], noise_pred_out[1]


                    noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
                    noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1], dim=1)

                    # default text guidance
                    noise_guidance = (noise_pred_text - noise_pred_uncond) * guidance_scale
                    if edit_momentum is None:
                        edit_momentum = torch.zeros_like(noise_guidance)

                    if self.sem_guidance is None:
                        self.sem_guidance = torch.zeros((len(timesteps), *noise_pred_text.shape))

                    if enable_edit_guidance:
                        noise_pred_edit_concepts = noise_pred_out[2:]
                        tmp = noise_pred_edit_concepts[0]
                        tmp, _ = tmp.split(model_input.shape[1], dim=1)

                        concept_weights = torch.zeros(
                            (len(tmp), noise_guidance.shape[0]),
                            device=edit_momentum.device,
                            dtype=noise_guidance.dtype,
                        )
                        noise_guidance_edit = torch.zeros(
                            (len(tmp), *noise_guidance.shape),
                            device=edit_momentum.device,
                            dtype=noise_guidance.dtype,
                        )
                        # noise_guidance_edit = torch.zeros_like(noise_guidance)
                        warmup_inds = []
                        for c, noise_pred_edit_concept in enumerate(noise_pred_edit_concepts):
                            noise_pred_edit_concept, _ = noise_pred_edit_concept.split(model_input.shape[1], dim=1)
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
                                    res=8,
                                    from_where=["up","down"],
                                    is_cross=True,
                                    select=self.text_cross_attention_maps.index(editing_prompt[c]),
                                )

                                attn_map = out[:, :, :num_edit_tokens[c]] # there is no startoftext

                                # average over all tokens
                                assert(attn_map.shape[2]==num_edit_tokens[c])
                                attn_map = torch.sum(attn_map, dim=2)

                                # gaussian_smoothing TODO
                                attn_map = F.pad(attn_map.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect")
                                attn_map = self.smoothing(attn_map).squeeze(0).squeeze(0)

                                # create binary mask
                                tmp = torch.quantile(attn_map.flatten(),edit_threshold_c)
                                attn_mask = torch.where(attn_map >= tmp, 1.0, 0.0)

                                # resolution must match latent space dimension
                                attn_mask = F.interpolate(
                                    attn_mask.unsqueeze(0).unsqueeze(0),
                                    noise_guidance_edit_tmp.shape[-2:] # 64,64
                                )[0,0,:,:]

                                if not use_intersect_mask:
                                    noise_guidance_edit_tmp = noise_guidance_edit_tmp * attn_mask
                            
                            if use_intersect_mask:
                                noise_guidance_edit_tmp_quantile = torch.abs(noise_guidance_edit_tmp)
                                noise_guidance_edit_tmp_quantile = torch.sum(noise_guidance_edit_tmp_quantile, dim=1, keepdim=True)
                                noise_guidance_edit_tmp_quantile = noise_guidance_edit_tmp_quantile.repeat(1,noise_guidance_edit_tmp.shape[1],1,1)

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

                                sega_mask = torch.where(
                                    noise_guidance_edit_tmp_quantile >= tmp[:, :, None, None],
                                    torch.ones_like(noise_guidance_edit_tmp),
                                    torch.zeros_like(noise_guidance_edit_tmp),
                                )

                                intersect_mask = sega_mask * attn_mask
                                noise_guidance_edit_tmp = noise_guidance_edit_tmp * intersect_mask

                            elif not use_cross_attn_mask:
                                # calculate quantile
                                noise_guidance_edit_tmp_quantile = torch.abs(noise_guidance_edit_tmp)
                                noise_guidance_edit_tmp_quantile = torch.sum(noise_guidance_edit_tmp_quantile, dim=1, keepdim=True)
                                noise_guidance_edit_tmp_quantile = noise_guidance_edit_tmp_quantile.repeat(1,noise_guidance_edit_tmp.shape[1],1,1)

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

                                noise_guidance_edit_tmp = torch.where(
                                    noise_guidance_edit_tmp_quantile >= tmp[:, :, None, None],
                                    noise_guidance_edit_tmp,
                                    torch.zeros_like(noise_guidance_edit_tmp),
                                )

                            noise_guidance_edit[c, :, :, :, :] = noise_guidance_edit_tmp

                        warmup_inds = torch.tensor(warmup_inds).to(self.device)
                        if len(noise_pred_edit_concepts) > warmup_inds.shape[0] > 0:
                            concept_weights = concept_weights.to("cpu")  # Offload to cpu
                            noise_guidance_edit = noise_guidance_edit.to("cpu")

                            concept_weights_tmp = torch.index_select(concept_weights.to(self.device), 0, warmup_inds)
                            concept_weights_tmp = torch.where(
                                concept_weights_tmp < 0, torch.zeros_like(concept_weights_tmp), concept_weights_tmp
                            )
                            concept_weights_tmp = concept_weights_tmp / concept_weights_tmp.sum(dim=0)
                            # concept_weights_tmp = torch.nan_to_num(concept_weights_tmp)

                            noise_guidance_edit_tmp = torch.index_select(
                                noise_guidance_edit.to(self.device), 0, warmup_inds
                            )
                            noise_guidance_edit_tmp = torch.einsum(
                                "cb,cbijk->bijk", concept_weights_tmp, noise_guidance_edit_tmp
                            )
                            noise_guidance_edit_tmp = noise_guidance_edit_tmp
                            noise_guidance = noise_guidance + noise_guidance_edit_tmp

                            self.sem_guidance[i] = noise_guidance_edit_tmp.detach().cpu()

                            del noise_guidance_edit_tmp
                            del concept_weights_tmp
                            concept_weights = concept_weights.to(self.device)
                            noise_guidance_edit = noise_guidance_edit.to(self.device)

                        concept_weights = torch.where(
                            concept_weights < 0, torch.zeros_like(concept_weights), concept_weights
                        )

                        concept_weights = torch.nan_to_num(concept_weights)

                        noise_guidance_edit = torch.einsum("cb,cbijk->bijk", concept_weights, noise_guidance_edit)

                        noise_guidance_edit = noise_guidance_edit + edit_momentum_scale * edit_momentum

                        edit_momentum = edit_mom_beta * edit_momentum + (1 - edit_mom_beta) * noise_guidance_edit

                        if warmup_inds.shape[0] == len(noise_pred_edit_concepts):
                            #print(noise_guidance.device, noise_guidance_edit.device)
                            noise_guidance = noise_guidance + noise_guidance_edit
                            self.sem_guidance[i] = noise_guidance_edit.detach().cpu()

                    noise_pred = noise_pred_uncond + noise_guidance
                    noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

                if self.scheduler.config.variance_type not in ["learned", "learned_range"]:
                    noise_pred, _ = noise_pred.split(model_input.shape[1], dim=1)

                # compute the previous noisy sample x_t -> x_t-1
                if use_ddpm:
                    idx = t_to_idx[int(t)]
                    intermediate_images = self.scheduler.step(
                        noise_pred, t, intermediate_images, variance_noise=zs[idx], **extra_step_kwargs, return_dict=False
                    )[0]
                else:
                    intermediate_images = self.scheduler.step(
                        noise_pred, t, intermediate_images, **extra_step_kwargs, return_dict=False
                    )[0]

                if use_cross_attn_mask:
                    # step callback
                    store_step = i in attn_store_steps
                    if store_step:
                        print(f"storing attention for step {i}")
                    self.attention_store.between_steps(store_step)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, intermediate_images)

        image = intermediate_images

        if output_type == "pil":
            # 8. Post-processing
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()

            # 9. Run safety checker
            image, nsfw_detected, watermark_detected = self.run_safety_checker(image, device, prompt_embeds.dtype)

            # 10. Convert to PIL
            image = self.numpy_to_pil(image)

            # 11. Apply watermark
            if self.watermarker is not None:
                image = self.watermarker.apply_watermark(image, self.unet.config.sample_size)
        elif output_type == "pt":
            nsfw_detected = None
            watermark_detected = None

            if hasattr(self, "unet_offload_hook") and self.unet_offload_hook is not None:
                self.unet_offload_hook.offload()
        else:
            # 8. Post-processing
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()

            # 9. Run safety checker
            image, nsfw_detected, watermark_detected = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, nsfw_detected, watermark_detected)

        return IFPipelineOutput(images=image, nsfw_detected=nsfw_detected, watermark_detected=watermark_detected)


    @torch.no_grad()
    def invert(self,
                image_path: str,
                source_prompt: str = "",
                source_guidance_scale = 3.5,
                num_inversion_steps: int = 100,
                skip: float = .15,
                eta: float = 1.0,
                generator: Optional[torch.Generator] = None
        ):
        """
        Inverts a real image according to Algorihm 1 in https://arxiv.org/pdf/2304.06140.pdf,
        based on the code in https://github.com/inbarhub/DDPM_inversion

        returns:
        zs - noise maps
        xts - intermediate inverted latents
        """
        self.eta = eta
        assert(self.eta > 0)

        device = self._execution_device
        dtype = self.text_encoder.dtype

        train_steps = self.scheduler.config.num_train_timesteps
        timesteps = torch.from_numpy(
            np.linspace(train_steps - skip * train_steps - 1, 0, num_inversion_steps).astype(np.int64)).to(self.device)
        #timesteps += self.scheduler.config.steps_offset

        self.num_inversion_steps = timesteps.shape[0]
        self.scheduler.num_inference_steps = timesteps.shape[0]
        self.scheduler.timesteps = timesteps
        #print(timesteps)

        reset_dpm(self.scheduler)

        # Reset attn processor, we do not want to store attn maps during inversion
        self.unet.set_attn_processor(AttnAddedKVProcessor())

        # 1. get embeddings
        text_embeddings, uncond_embedding, _, _ = self.encode_prompt(source_prompt)
        prompt_embeds = torch.cat([uncond_embedding, text_embeddings])

        # 2. open image
        image = self.crop(image_path)
        x0 = self.preprocess_image(image)
        x0 = x0.to(device=device, dtype=dtype)
        self.batch_size = x0.shape[0]

        # 3. find zs and xts
        variance_noise_shape = (
            self.num_inversion_steps,
            self.batch_size,
            self.unet.config.in_channels,
            self.unet.sample_size,
            self.unet.sample_size)

        # intermediate latents
        t_to_idx = {int(v):k for k,v in enumerate(timesteps)}
        xts = torch.zeros(size=variance_noise_shape, device=device, dtype=uncond_embedding.dtype)

        for t in reversed(timesteps):
            idx = self.num_inversion_steps-t_to_idx[int(t)] - 1
            noise = randn_tensor(shape=x0.shape, generator=generator, device=device, dtype=x0.dtype)
            xts[idx] = self.scheduler.add_noise(x0, noise, t)
        xts = torch.cat([x0.unsqueeze(0), xts], dim=0)

        # noise maps
        zs = torch.zeros(size=variance_noise_shape, device=device, dtype=uncond_embedding.dtype)

        if hasattr(self, "text_encoder_offload_hook") and self.text_encoder_offload_hook is not None:
            self.text_encoder_offload_hook.offload()

        for t in tqdm(timesteps):
            idx = self.num_inversion_steps-t_to_idx[int(t)]-1

            # 1. predict noise residual
            xt = xts[idx+1]
            model_input = torch.cat([xt] * 2)
            noise_pred = self.unet(
                model_input,
                timestep=t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=None,
                return_dict=False
            )[0]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
            noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1], dim=1)

            noise_pred = noise_pred_uncond + (noise_pred_text - noise_pred_uncond) * source_guidance_scale

            xtm1 = xts[idx]
            z, xtm1_corrected = compute_noise(self.scheduler, xtm1, xt, t, noise_pred, eta)
            zs[idx] = z

            # correction to avoid error accumulation
            xts[idx] = xtm1_corrected

        # TODO: I don't think that the noise map for the last step should be discarded ?!
        # if not zs is None:
        #     zs[-1] = torch.zeros_like(zs[-1])

        self.init_images = xts[-1].expand(1, -1, -1, -1)
        zs = zs.flip(0)
        self.zs = zs

        if hasattr(self, "unet_offload_hook") and self.unet_offload_hook is not None:
            self.unet_offload_hook.offload()

        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        return zs, xts


# Copied from pipelines.StableDiffusion.CycleDiffusionPipeline.compute_noise
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
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * noise_pred

    # modifed so that updated xtm1 is returned as well (to avoid error accumulation)
    mu_xt = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

    if variance > 0.0:
        noise = (prev_latents - mu_xt) / (variance ** (0.5) * eta)
    else:
        noise = torch.Tensor([0.0]).to(latents.device)

    return noise, mu_xt + ( eta * variance ** 0.5 )*noise

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

        if sigma > 0.0:
            noise = (prev_latents - mu_xt) / sigma
        else:
            noise = torch.Tensor([0.0]).to(sample.device)

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

        if sigma > 0.0:
            noise = (prev_latents - mu_xt) / sigma
        else:
            noise = torch.Tensor([0.0]).to(sample.device)

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
