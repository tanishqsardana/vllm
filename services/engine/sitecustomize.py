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
    import importlib.abc
    import sys

    TARGET_MODULE = "vllm.model_executor.model_loader.weight_utils"

    def _patch_vllm_disabled_tqdm(module) -> None:
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
                    self.disable = True

            module.DisabledTqdm = PatchedDisabledTqdm
            module._codex_tqdm_patched = True
        except Exception:
            # Defer silently; module may not be fully ready yet.
            return

    class _PatchLoader(importlib.abc.Loader):
        def __init__(self, wrapped_loader):
            self._wrapped_loader = wrapped_loader

        def create_module(self, spec):
            if hasattr(self._wrapped_loader, "create_module"):
                return self._wrapped_loader.create_module(spec)
            return None

        def exec_module(self, module):
            self._wrapped_loader.exec_module(module)
            _patch_vllm_disabled_tqdm(module)

    class _PatchFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != TARGET_MODULE:
                return None

            for finder in sys.meta_path:
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec and spec.loader:
                    spec.loader = _PatchLoader(spec.loader)
                    return spec
            return None

    if not any(type(f).__name__ == "_PatchFinder" for f in sys.meta_path):
        sys.meta_path.insert(0, _PatchFinder())

    _patch_vllm_disabled_tqdm(sys.modules.get(TARGET_MODULE))
except Exception:
    # Keep startup resilient if import hook cannot be installed.
    pass
