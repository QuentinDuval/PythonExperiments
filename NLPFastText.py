from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import fastText


"""
Taking example of
https://github.com/facebookresearch/fastText/blob/master/python/doc/examples/train_supervised.py
https://fasttext.cc/docs/en/supervised-tutorial.html
"""


def print_results(number_of_samples, precision, recall):
    print("N\t" + str(number_of_samples))
    print("P@{}\t{:.3f}".format(1, precision))
    print("R@{}\t{:.3f}".format(1, recall))


def train_fast_text():
    train_data = 'nlp_fasttext_train.txt'
    valid_data = 'nlp_fasttext_valid.txt'

    model = fastText.train_supervised(
        input=train_data, epoch=10, lr=1.0, wordNgrams=2, verbose=2, minCount=1
    )
    print_results(*model.test(valid_data))
    model.save_model("nlp_fasttext_model.bin")


def load_fast_text():
    model = fastText.load_model("nlp_fasttext_model.bin")
    print(model.predict("I love football"))


train_fast_text()
load_fast_text()
