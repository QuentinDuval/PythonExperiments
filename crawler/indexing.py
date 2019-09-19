from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import QueryParser


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
