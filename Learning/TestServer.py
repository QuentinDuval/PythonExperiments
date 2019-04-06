from flask import Flask, render_template, request, jsonify
from functools import lru_cache
from Learning.Classifier3 import *
from Learning.WordEmbeddings import *


app = Flask(__name__)


# TODO - add debugging facilities (show training curve, show parsing result, show all percentages...)


@lru_cache(maxsize=None)
def get_classification_model():
    return load_best_model()


@lru_cache(maxsize=None)
def get_embedding_search_index():
    embeddings = WordEmbeddings.load_from(model_path='resources/unsupervised_model.bin')
    return WordIndex(embeddings)


@app.route("/devoxx")
def open_gui():
    return render_template("TestAI.html",
                           host="127.0.0.1",
                           port=request.environ['SERVER_PORT'])


@app.route("/devoxx/guess", methods=['GET', 'POST'])
def guess_changelist_type():
    content = request.data
    if not content:
        return "Empty fix description!"
    else:
        content = request.data.decode("utf-8")
        return str(get_classification_model().predict(content))


@app.route("/devoxx/neighbors", methods=['GET', 'POST'])
def get_word_neighbors():
    content = request.data
    if not content:
        return "Empty fix description!"
    else:
        content = request.data.decode("utf-8")
        neighbors = get_embedding_search_index().neighbors(content)
        neighbors = [{"key": n, "value": 20} for n in neighbors]
        return jsonify({"neighbors": neighbors})


@app.route("/devoxx/generate", methods=['GET'])
def generate_commit():
    return "empty commit message to test" # TODO


if __name__ == '__main__':
    # get_classification_model()
    # get_embedding_search_index()

    # For debugging purposes, run on the default port
    # For production, use "flask run --port=80"
    app.run(host='0.0.0.0')

