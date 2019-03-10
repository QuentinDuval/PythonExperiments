from annoy import AnnoyIndex
import fastText
import nltk

# https://fasttext.cc/docs/en/english-vectors.html


def clean_text():
    with open('WordEmbeddingsInput.txt', 'r') as f_in:
        with open('WordEmbeddingsOutput.txt', 'w') as f_out:
            for line in f_in:
                words = nltk.word_tokenize(line)
                words = [w.lower() for w in words]
                out_line = " ".join(words)
                f_out.write(out_line + "\n")


def promixity_queries():
    embedding_size = 20

    model = fastText.train_unsupervised(input='WordEmbeddingsOutput.txt', dim=embedding_size, model='skipgram')
    known_words = set(model.get_words())

    '''
    for word in model.get_words():
        word_id = model.get_word_id(word)
        embedding = model.get_word_vector(word)
        print(word, word_id, embedding)
    '''

    '''
    print("-" * 50)
    print(model.get_input_matrix())     # The embedding matrix apparently
    print("-" * 50)
    print(model.get_output_matrix())    # No idea what this is
    print("-" * 50)
    '''

    # To query for nearest neighbors

    index = AnnoyIndex(embedding_size)
    for word in model.get_words():
        index.add_item(model.get_word_id(word), model.get_word_vector(word))
    index.build(5)
    # index.save('annoy.ann')

    # index.load('annoy.ann')
    while True:
        query = input("query>").split(' ')

        # analogy query
        if len(query) == 3:
            a, b, c = query
            if a in known_words and b in known_words and c in known_words:
                a, b, c = [model.get_word_vector(w) for w in [a, b, c]]
                result = index.get_nns_by_vector(vector=c + (b - a), n=10)
                print([model.get_words()[word_i] for word_i in result])
            else:
                print('unknown words', {a, b, c} - known_words)

        # proximity query
        elif len(query) == 1:
            v = model.get_word_vector(query[0])
            result = index.get_nns_by_vector(vector=v, n=10)
            print([model.get_words()[word_i] for word_i in result])


# clean_text()
# promixity_queries()
