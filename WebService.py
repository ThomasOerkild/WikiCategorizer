from flask import Flask, request, jsonify
from doc2vecmodel import Doc2VecModel
import json
import os

app = Flask(__name__)

HOME = os.path.expanduser("~")
MODEL_PATH = os.path.join(HOME, "Documents", "my_doc2vec_model_trained")

print("Loading Doc2Vec model...")
d2v = Doc2VecModel(MODEL_PATH)
print("Finished loading model.")

@app.route('/api/mostsimilar/', methods=['POST'])
def most_similar():
    text = request.form['text']
    n = int(request.form['n'])
    titles, scores = d2v.infer(text, n)
    return jsonify({'titles': titles, 'scores': scores})

if __name__ == '__main__':
    app.run(host='localhost', debug=False, threaded=True, use_reloader=False)