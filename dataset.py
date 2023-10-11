
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

class TextDataset(Dataset):
    def __init__(self, text_datasets, tokenizer):
        super().__init__()
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets)
        self.left_pad = True
        self.pad_idx = tokenizer.pad_token_id

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with torch.no_grad():
            out_kwargs = {}
            out_kwargs['input_ids'] = np.array(
                self.text_datasets[idx]['input_ids'])
            out_kwargs['attention_mask'] = np.array(
                self.text_datasets[idx]['attention_mask'])
            out_kwargs['prompt_len'] = np.array(
                len(self.text_datasets[idx]['input_ids']))
            return out_kwargs

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        def merge(key, pad_token):
            return collate_tokens([s[key] for s in samples], pad_token,
                                  self.left_pad)

        input_ids = merge("input_ids", self.pad_idx)
        attention_mask = merge("attention_mask", 0)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_len": input_ids.size(1)
        }
        return batch


def collate_tokens(
    values,
    pad_idx,
    left_pad=False,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    values = [torch.LongTensor(v) for v in values]
    size = max(v.size(0) for v in values)
    batch_size = len(values)
    res = values[0].new(batch_size, size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res

