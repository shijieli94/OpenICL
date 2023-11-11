import collections
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy


def apply_to_sample(f, sample):
    if hasattr(sample, "__len__") and len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, collections.OrderedDict):
            # OrderedDict has attributes that needs to be preserved
            od = collections.OrderedDict((key, _apply(value)) for key, value in x.items())
            od.__dict__ = x.__dict__
            return od
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x

    return _apply(sample)


class ListWrapper:
    def __init__(self, data: List[Any]):
        self.data = data

    def to(self, device):
        def _move_to_device(tensor):
            return tensor.to(device=device, non_blocking=True)

        return apply_to_sample(_move_to_device, self.data)


def ignore_pad_dict(features):
    res_dict = {}
    if "metadata" in features[0]:
        res_dict["metadata"] = ListWrapper([x.pop("metadata") for x in features])
    return res_dict


@dataclass
class DataCollatorWithPaddingAndCuda:
    tokenizer: PreTrainedTokenizerBase
    device: object = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = 3000
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> BatchEncoding:
        res_dict = ignore_pad_dict(features)

        has_labels = "labels" in features[0]
        if has_labels:
            labels = [{"input_ids": x.pop("labels")} for x in features]
            labels = self.tokenizer.pad(
                labels,
                padding=True,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_attention_mask=True,
                return_tensors="pt",
                verbose=False,
            )

        # print(features)
        batch = self.tokenizer.pad(
            features,
            padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt",
            verbose=False,
        )

        if has_labels:
            batch["labels"] = labels.input_ids

        batch.update(res_dict)

        if self.device:
            batch = batch.to(self.device)

        return batch
