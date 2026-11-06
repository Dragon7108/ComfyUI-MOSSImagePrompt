"""ComfyUI-MOSSImagePrompt: MOSS-Image prompt + reference images with a separate weight for each image and the text."""
from .nodes import MOSSImagePrompt, MOSSImagePromptLegacy

NODE_CLASS_MAPPINGS = {"MOSSImagePrompt": MOSSImagePrompt,
                       "NeuronSteerMOSSImagePrompt": MOSSImagePromptLegacy}
NODE_DISPLAY_NAME_MAPPINGS = {"MOSSImagePrompt": "MOSS Image Prompt",
                              "NeuronSteerMOSSImagePrompt": "MOSS Image Prompt (legacy key)"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
