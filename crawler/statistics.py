from crawler.parsing import NltkTokenizer, list_files_in_folder, stop_words


def count_words(folder: str):
    word_groups = {}
    word_bigrams = {}
    tokenizer = NltkTokenizer()
    for file_name in list_files_in_folder(folder):
        with open(file_name, 'r', encoding='utf-8') as file:
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
    if word not in stop_words:
        print(word, count)

print("-" * 50)

for word, count in sorted(bigrams.items(), key=lambda p: p[1], reverse=True):
    print(word, count)
