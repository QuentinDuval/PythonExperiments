from torch.utils.data import Dataset
import random


class SplitDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, item):
        return self.xs[item], self.ys[item]

    def split(self, ratio: float):
        indices = list(range(len(self)))
        split_point = int(ratio * len(indices))
        random.shuffle(indices)
        lhs, rhs = indices[:split_point], indices[split_point:]
        lhs_xs, rhs_xs = [self.xs[i] for i in lhs], [self.xs[i] for i in rhs]
        lhs_ys, rhs_ys = [self.ys[i] for i in lhs], [self.ys[i] for i in rhs]
        return SplitDataset(lhs_xs, lhs_ys), SplitDataset(rhs_xs, rhs_ys)
