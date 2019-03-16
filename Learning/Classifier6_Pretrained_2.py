from Learning.Classifier6 import *
from Learning.WordEmbeddings import *

# TODO - try with setting the initial weights, and not freezing the weights


def test_model_6_pretrained_filling(split_seed=None):
    training_corpus = CommitMessageCorpus.from_split('train')
    test_corpus = CommitMessageCorpus.from_split('test')

    vectorizer = EmbeddingRnnVectorizer.from_corpus(training_corpus, NltkTokenizer(), min_freq=2, max_length=50)
    vocab_len = vectorizer.get_vocabulary_len()

    model = RnnClassifier(vocabulary_len=vocab_len, embedding_size=20, nb_classes=4)

    # fill the embeddings
    # TODO - does not seem to improve anything...
    # TODO - another idea is to use it to improve the "unknown" words, but not that awesome... and we cheat
    pretrained_embeddings = WordEmbeddings.load_from(model_path='resources/unsupervised_model.bin')
    pretrained_embeddings.transfer_learning_to(vectorizer.vocabulary, model.embed)

    predictor = Predictor(model=model, vectorizer=vectorizer, with_gradient_clipping=True, split_seed=split_seed)
    predictor.fit(training_corpus=training_corpus, learning_rate=0.01, weight_decay=0.0)
    predictor.evaluate(test_corpus=test_corpus)


"""
Training (max): 3896/4244 (91.80018850141376%)
Validation (max): 365/472 (77.33050847457628%)
------------------------------
Accuracy: 73.19587628865979 %
"""

# test_model_6_pretrained_filling(split_seed=None)
