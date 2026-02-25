"""Runtime compatibility shims for tokenizer API differences.

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
