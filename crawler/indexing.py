from typing import List

from crawler.parsing import NltkTokenizer, list_files_in_folder

from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import QueryParser

import os


def test_whoosh():
    schema = Schema(title=ID(stored=True), content=TEXT(stored=True))

    ix = create_in("test_index_dir", schema)
    writer = ix.writer()
    writer.add_document(title=u"a", content=u"This is the first document we've added!")
    writer.add_document(title=u"b", content=u"The second one is even more interesting!")
    writer.add_document(title=u"c", content=u"This is the third document, better than the first!")
    writer.commit()

    writer = ix.writer()
    writer.delete_by_term('title', 'c')
    writer.commit()

    with ix.searcher() as searcher:
        query = QueryParser("content", ix.schema).parse("first")
        results = searcher.search(query)
        for result in results:
            print(result['title'])


class InvIndex:
    """
    Maintains an inverted index of documents (blog posts here) from words to documents that contain the words.
    """

    def __init__(self):
        self.schema = Schema(title=ID(stored=True), content=TEXT)
        self.ix = create_in("post_indexing", self.schema)

    def add_document(self, file_path: str):
        writer = self.ix.writer()
        folder_name, file_name = os.path.split(file_path)
        with open(file_path, 'r', encoding='utf-8') as file:
            writer.add_document(title=file_name, content=file.read())
        writer.commit()

    def get_topic(self, document_title: str) -> List[str]:
        """
        Return the titles of the document that are the most related to the document_title
        """
        # TODO
        #   - get the words in the document, and look at the words in the posting list of documents
        #   - check if the document is among the most representative for each word (and keep this word)
        pass

    def get_documents(self, words: List[str]) -> List[str]:
        """
        Return the list of title of the document that are the most related to the words given as input
        """
        with self.ix.searcher() as searcher:
            query = QueryParser("content", self.ix.schema).parse(words)
            results = searcher.search(query)
            documents = []
            for result in results:
                documents.append(result['title'])
            return result

    def get_common_expressions(self) -> List[str]:
        """
        Use TF-IDF / correlation measure to return the terms that are the most associated together
        """
        pass  # TODO

    def get_most_common_next_words(self) -> List[str]:
        pass


inv_index = InvIndex()
for file_path in list_files_in_folder("posts/fluentcpp"):
    inv_index.add_document(file_path)

print(inv_index.get_documents(['expressive']))
