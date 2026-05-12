from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from transformers import PreTrainedTokenizerBase

class BaseFormatter(ABC):
    """
    Abstract base class for all formatters.

    Contract:
      - Callable: formatter(example, task, tokenizer=None) -> Dict
      - Subclasses implement .format(...); __call__ forwards to it.
    """

    def __call__(
        self,
        example: Dict[str, Any],
        task: str,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> Dict[str, Any]:
        return self.format(example, task, tokenizer)

    @abstractmethod
    def format(
        self,
        example: Dict[str, Any],
        task: str,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError