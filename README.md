# ComfyUI-MOSSImagePrompt

A copy of ComfyUI's **Text Encode MOSS-Image Edit** node (no-VAE path) that gives **each reference image its own strength slider**, plus one for the prompt's words.

## Why

MOSS-Image's text encoder (Qwen3.5) is a vision-language model: give it images without a VAE and it reads them alongside your prompt. Each image becomes about **144 tokens**, while a prompt might be 10, so the images drown out your words. Core's single `image_strength` turns *all* images down together. This node lets you weight them separately and choose which one leads.

Base MOSS was never trained on reference images, so this works as a **semantic image prompt**, not a style copier. It carries over what's *in* the picture (subjects, themes, a vibe), usually rendered in MOSS's own look. Style transfers only when style is the image's dominant trait (e.g. a watercolor wash). Several images blend their concepts.

## Install

**Via ComfyUI-Manager:** "Install via Git URL" → `https://github.com/Dragon7108/ComfyUI-MOSSImagePrompt`

**Manual:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Dragon7108/ComfyUI-MOSSImagePrompt
```
Restart ComfyUI. No extra dependencies.

**Requires** a ComfyUI version that includes MOSS-Image's `context_attention_bias` (PR #17320, the one that added `image_strength` to the core node). On older versions the sliders silently do nothing.

## Usage

Use it in place of **Text Encode MOSS-Image Edit**, on the **positive** prompt:

```
Load CLIP (MOSS-Image text encoder) ─┐
Load Image ── image_1 ───────────────┼─ MOSS Image Prompt ── positive ── KSampler
Load Image ── image_2 ───────────────┘
```

Keep the negative on a plain **CLIP Text Encode**. There's no VAE input.

Found under `model/conditioning/moss_image`, next to the core node.

## Settings

| Input | Default | What it does |
|---|---|---|
| `prompt` | | Your text prompt. |
| `text_strength` | 1.0 | Weight of the prompt's words. Above 1 (2–3) lets the words push back against the images. |
| `vision_size` | 384 | Each image is downscaled to at most this many pixels squared before the encoder reads it (core uses 384). Bigger = more tokens = more detail and more pull; lower its strength to match. |
| `image_1`…`image_4` | | Optional reference images. Only the first frame of a batch is used. |
| `strength_1`…`strength_4` | 0.05 | Each image's weight. **1** = unweighted (swamps the prompt), **~0.01–0.05** = a hint (the sweet spot), **0** = image left out. |

The **`info`** output (wire it to a Preview Any) lists each image's token count and the bias applied to every part.

**Tips:** start images around 0.02–0.05 and nudge from there. Strengths are relative multipliers, not percentages: compare two images' strengths directly, and use `text_strength` to balance images against text.

## How it works

ComfyUI's MOSS-Image model accepts `context_attention_bias`: one value per conditioning token, added to that token's attention logits in the diffusion model. Core's node puts `log(image_strength)` on every vision token. This node fills the bias per segment instead: `log(strength_i)` on image *i*'s tokens, `log(text_strength)` on the prompt's words, and 0 on the chat template and the `Picture N:` labels. At matching settings the output is pixel-identical to the core node.

## License

[GPL-3.0](LICENSE) — ComfyUI is GPL-3.0 and this node imports its APIs, so it matches the ecosystem.
