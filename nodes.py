"""MOSS Image Prompt: core's "Text Encode MOSS-Image Edit" (no-VAE path) with a separate weight per image and for the
prompt's own words.

MOSS's text encoder (Qwen3.5) is a vision-language model. Images given to it become ~144 vision tokens each (at 384²),
which swamp a short prompt (ComfyUI PR #17320). Core's `image_strength` turns *all* vision tokens down by the same amount:
it adds log(strength) to their attention logits in the DiT through `context_attention_bias`, a bias with one value per
conditioning token (`comfy/ldm/moss_image/model.py`). This node fills that bias per segment instead:
- log(strength_i) on image i's vision tokens
- log(text_strength) on the prompt's own words
- 0 on the chat template and the "Picture N:" labels

No VAE, so the images only reach the text encoder: base MOSS was never trained on reference latents. At low strength
the images act as a semantic image prompt (subject/theme, and style when style is the image's dominant trait).
"""
import math

import torch

import comfy.text_encoders.qwen_vl
import comfy.utils
import node_helpers

# Qwen3.5 tokenizer ids for the chat template MOSS wraps every prompt in:
# <|im_start|>user\n {prompt} <|im_end|>\n<|im_start|>assistant\n<think>\n
IM_START, IM_END, NEWLINE = 248045, 248046, 198

SLOTS = 4


class MOSSImagePrompt:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, SLOTS + 1):
            optional["image_%d" % i] = ("IMAGE",)
            optional["strength_%d" % i] = ("FLOAT", {"default": 0.05, "min": 0.0, "max": 10.0, "step": 0.005, "tooltip":
                "Weight of image %d's vision tokens. 1 = unweighted (swamps the prompt), ~0.01-0.05 = a hint, "
                "0 = leave the image out." % i})
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "text_strength": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.05, "tooltip":
                    "Weight of the prompt's own words. Above 1 lets the words hold their own against the images."}),
                "vision_size": ("INT", {"default": 384, "min": 64, "max": 1024, "step": 32, "tooltip":
                    "Each image is downscaled to at most this many pixels squared before the vision tower (core: 384). "
                    "Bigger = more vision tokens = more detail, and more weight. Also slower: every sampling step "
                    "attends to all of them (two images at 1024 cost ~25% more time than at 384)."}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "info")
    FUNCTION = "encode"
    CATEGORY = "model/conditioning/moss_image"
    DESCRIPTION = ("MOSS-Image prompt with up to 4 reference images going only to the text encoder, each with its "
                   "own weight, plus a weight for the prompt's words. Positive prompt only; no VAE, no LoRA needed.")

    def encode(self, clip, prompt, text_strength, vision_size, **kw):
        images, skipped = [], []
        for i in range(1, SLOTS + 1):
            img, s = kw.get("image_%d" % i), kw.get("strength_%d" % i, 0.05)
            if img is not None and s > 0:
                images.append((i, img[:1], s))
            elif img is not None:
                skipped.append(i)

        vl = []
        for _, img, _ in images:
            scale = min(1.0, math.sqrt(vision_size * vision_size / (img.shape[1] * img.shape[2])))
            vl.append(comfy.utils.common_upscale(img.movedim(-1, 1), round(img.shape[2] * scale), round(img.shape[1] * scale),
                                                 "area", "disabled").movedim(1, -1)[:, :, :, :3])
        text = "".join("Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(n + 1) for n in range(len(vl))) + prompt
        tokens = clip.tokenize(text, images=vl)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        seq = next(iter(tokens.values()))[0]
        ids = [None if isinstance(t[0], dict) else t[0] for t in seq]
        img_pos = [p for p, t in enumerate(ids) if t is None]
        try:
            end = len(ids) - 1 - ids[::-1].index(IM_END)
            # words start after the last image's <|vision_end|>, or after "<|im_start|>user\n" with no images
            start = img_pos[-1] + 2 if img_pos else ids.index(NEWLINE, ids.index(IM_START)) + 1
        except ValueError:
            raise ValueError("MOSS Image Prompt: couldn't find the prompt inside the chat template. Use a MOSS-Image "
                             "text encoder, and don't start the prompt with your own <|im_start|> template.")

        bias, info, k = [], [], 0
        tw = math.log(text_strength)
        for p, t in enumerate(seq):
            if ids[p] is None:
                slot, _, s = images[k]
                _, grid = comfy.text_encoders.qwen_vl.process_qwen2vl_images(t[0]["data"], patch_size=16)
                n = int(grid[0, 1] * grid[0, 2]) // 4  # Qwen3.5 merges 2x2 patches of 16 px into one token
                bias += [math.log(s)] * n
                info.append("image_{} (Picture {}): {} vision tokens, strength {:g} (bias {:+.2f})".format(
                    slot, k + 1, n, s, math.log(s)))
                k += 1
            else:
                bias.append(tw if start <= p < end else 0.0)
        info += ["image_{}: left out (strength 0)".format(i) for i in skipped]
        info.append("prompt: {} tokens, strength {:g} (bias {:+.2f})".format(max(0, end - start), text_strength, tw))

        n_ctx = conditioning[0][0].shape[1]
        if len(bias) != n_ctx:
            raise ValueError("MOSS Image Prompt: bias covers {} tokens but the conditioning has {}".format(len(bias), n_ctx))
        if any(b != 0.0 for b in bias):
            conditioning = node_helpers.conditioning_set_values(conditioning, {"context_attention_bias": torch.tensor([bias])})
        n_tpl = sum(1 for p, t in enumerate(ids) if t is not None and not start <= p < end)
        info.append("template + labels: {} tokens (bias +0.00)".format(n_tpl))
        info.append("total context tokens: {}".format(n_ctx))
        return (conditioning, "\n".join(info))


class MOSSImagePromptLegacy(MOSSImagePrompt):
    """The same node under its original key (it started out in the NeuronSteer pack), so workflows saved with it still
    load. Hidden from the node search."""
    DEPRECATED = True
