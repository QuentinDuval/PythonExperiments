from annoy import AnnoyIndex
import fastText

# https://fasttext.cc/docs/en/english-vectors.html

embedding_size = 10

model = fastText.train_unsupervised(input='WordEmbeddingsInput.txt', dim=embedding_size)
for word in model.get_words():
    word_id = model.get_word_id(word)
    embedding = model.get_word_vector(word)
    print(word, word_id, embedding)


# To query for nearest neighbors

index = AnnoyIndex(embedding_size)
for word in model.get_words():
    index.add_item(model.get_word_id(word), model.get_word_vector(word))
index.build(10)
# index.save('annoy.ann')

# index.load('annoy.ann')
while True:
    a, b, c = input("query>").split(' ')
    a, b, c = [model.get_word_vector(w) for w in [a, b, c]]
    result = index.get_nns_by_vector(vector=c + (b - a), n=10)
    print([model.get_words()[word_i] for word_i in result])
