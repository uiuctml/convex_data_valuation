from typing import List, Dict, Any, Optional
from transformers import PreTrainedTokenizerBase


class BigMathRLGSM8KGoogleTransRLGRPOCollator:
    """
    Identity collator for GRPO training.

    Returns the batch as-is (list of dicts) to bypass PyTorch's default
    collation which would attempt to stack items into tensors. GRPO
    processes each item individually, so we preserve the list structure.
    """

    def __init__(self, tokenizer: Optional[PreTrainedTokenizerBase] = None):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Identity function - preserves list-of-dicts structure for GRPO
        return batch