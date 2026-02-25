"""Runtime compatibility shims for dependency API differences.

Loaded automatically by Python when present on sys.path as `sitecustomize`.
"""

try:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):

        @property
        def all_special_tokens_extended(self):
            base = list(getattr(self, "all_special_tokens", []) or [])
            extra = list(getattr(self, "additional_special_tokens", []) or [])
            merged = []
            for token in base + extra:
                if token not in merged:
                    merged.append(token)
            return merged

        PreTrainedTokenizerBase.all_special_tokens_extended = all_special_tokens_extended
except Exception:
    # Keep startup resilient if transformers isn't imported yet or changes APIs.
    pass

try:
    import builtins
    import sys

    _orig_import = builtins.__import__

    def _patch_vllm_disabled_tqdm() -> None:
        module = sys.modules.get("vllm.model_executor.model_loader.weight_utils")
        if module is None:
            return
        if getattr(module, "_codex_tqdm_patched", False):
            return

        try:
            from tqdm.asyncio import tqdm_asyncio

            class PatchedDisabledTqdm(tqdm_asyncio):
                """Avoid duplicate `disable` kwarg from mixed vLLM/hf-hub versions."""

                def __init__(self, *args, **kwargs):
                    kwargs.pop("disable", None)
                    super().__init__(*args, **kwargs, disable=True)

            module.DisabledTqdm = PatchedDisabledTqdm
            module._codex_tqdm_patched = True
        except Exception:
            # Defer silently; module may not be fully ready yet.
            return

    def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        mod = _orig_import(name, globals, locals, fromlist, level)
        _patch_vllm_disabled_tqdm()
        return mod

    builtins.__import__ = _patched_import
except Exception:
    # Keep startup resilient if import hook cannot be installed.
    pass
