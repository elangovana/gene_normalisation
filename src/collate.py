import torch


def collate(batch):
    data = torch.cat([item[0].unsqueeze(dim=0) for item in batch], dim=0)
    target = torch.cat([item[1].unsqueeze(dim=0) for item in batch], dim=0)

    return [data, target]