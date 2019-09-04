import os

from crawler.parsing import NltkTokenizer


def list_files(folder: str):
    files = []
    for dir_path, dir_names, file_names in os.walk(folder):
        files.extend(file_names)
    return file_names


def count_words(folder: str):
    word_groups = {}
    word_bigrams = {}
    tokenizer = NltkTokenizer()
    for file_name in list_files(folder):
        with open(os.path.join(folder, file_name), 'r', encoding='utf-8') as file:
            for line in file:
                line_words = tokenizer.tokenize(line)
                prev = None
                for word in line_words:
                    word = word.lower().strip(".?!")
                    if word.isalpha():
                        word_groups[word] = word_groups.get(word, 0) + 1
                        if prev is not None:
                            word_bigrams[prev + " " + word] = word_bigrams.get(prev + " " + word, 0) + 1
                        prev = word
    return word_groups, word_bigrams


words, bigrams = count_words("posts/fluentcpp")
for word, count in sorted(words.items(), key=lambda p: p[1], reverse=True):
    print(word, count)

print("-" * 50)

for word, count in sorted(bigrams.items(), key=lambda p: p[1], reverse=True):
    print(word, count)
