import fastText

from Learning.Corpus import *
from Learning.Evaluation import *


# fast_text_model = "defect_classification_fasttext_model.bin"


def print_fast_text_results(number_of_samples, precision, recall):
    print("Number of inputs:", number_of_samples)
    print("Precision:", precision)
    print("Recall:", recall)


def prepare_fast_text():
    def transform(corpus, output):
        with open(output, 'w') as out:
            for x, y in corpus:
                y = CommitMessageCorpus.target_class_index(y)
                out.writelines(['__label__' + str(y) + ' ' + x, '\n'])

    training_corpus = CommitMessageCorpus.from_split('train')
    transform(training_corpus, "perforce_cl_train_fasttext.txt")


def train_fast_text():
    model = fastText.train_supervised(
        input='resources/perforce_cl_train_fasttext.txt',
        epoch=10,       # 5, 10 or 20... it all works the same.
        lr=0.5,         # 0.5 leads to better performance than 1.0 (77%)
        wordNgrams=2,   # 2 seems to be better than 1...
        verbose=2,
        minCount=1      # 1 seems to be better than 2...
    )
    print_fast_text_results(*model.test('perforce_cl_train_fasttext.txt'))
    # model.save_model(fast_text_model)
    return model


def test_fast_text():
    # model = fastText.load_model(fast_text_model)
    model = train_fast_text()
    test_corpus = CommitMessageCorpus.from_split('test')
    expected = [CommitMessageCorpus.target_class_index(y) for y in test_corpus.get_targets()]

    predicted = []
    for x, y in test_corpus:
        output = model.predict(x.strip())
        print(x, output)
        predicted.append(int(output[0][0][-1]))
    ConfusionMatrix(expected, predicted, CommitMessageCorpus.TARGET_CLASSES).show()


"""
------------------------------------------------------------------------------------------------------------------------
BASELINE ALGORITHM 3 (fast text + unsupervised learning)
(https://github.com/facebookresearch/fastText/blob/master/python/doc/examples/train_unsupervised.py)
------------------------------------------------------------------------------------------------------------------------
"""


def fast_text_unsupervised_learning():
    model = fastText.train_unsupervised(input='resources/perforce_cl_unsupervised.txt', model='skipgram')
    # model.save_model("defect_classification_fasttext_unsupservised_model.bin")
    return model


def test_fast_text_unsupervised_learning():
    # model = fastText.load_model("defect_classification_fasttext_unsupservised_model.bin")
    model = fast_text_unsupervised_learning()

    # TODO - you could use these vectors as pretrained vectors for the algorithm afterwards
    # TODO - use get_subwords as well :)

    print("Length of the vocabulary:", len(model.get_words()))
    print("Length of the output matrix:", len(model.get_output_matrix()))
    print("Size of the embedding:", len(model.get_output_matrix()[0]))

    print("Example of words:", model.get_words()[:50])

    """
    while True:
        word = input("Word?>")
        print(">", model.predict(word))
    """


# prepare_fast_text()
# test_fast_text()
# test_fast_text_unsupervised_learning()
