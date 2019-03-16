from torch.utils.data import Dataset

from Learning.Corpus import *
from Learning.Vectorizer import *
from Learning.Utils import *


class CommitMessageDataSet(Dataset):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    @classmethod
    def from_corpus(cls, corpus: CommitMessageCorpus, vectorizer: Vectorizer):
        xs = []
        ys = []
        for x, y in corpus:
            xs.append(vectorizer.vectorize(x))
            ys.append(corpus.target_class_index(y))
        return cls(xs, ys)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, index):
        return {'x': self.xs[index],
                'y': self.ys[index]}

    def split(self, ratio, seed=None):
        lhs, rhs = join_split(self.xs, self.ys, ratio, seed=seed)
        return CommitMessageDataSet(*lhs), CommitMessageDataSet(*rhs)

    def get_batch_number(self, batch_size):
        return len(self) // batch_size
