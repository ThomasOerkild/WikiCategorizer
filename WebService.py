from flask import Flask, request, jsonify
from doc2vecmodel import Doc2VecModel, NNSMethod
import json
import os

app = Flask(__name__)

HOME = os.path.expanduser("~")
MODEL_PATH = os.path.join(HOME, "Documents", "100vector_model", "my_doc2vec_model_trained")

METHOD_DICT = {'brute': NNSMethod.BRUTE, 
                'kdtree': NNSMethod.KD_TREE, 
                'annoy': NNSMethod.ANNOY }

print("Loading Doc2Vec model...")
d2v = Doc2VecModel(MODEL_PATH)
print("Finished loading model.")

@app.route('/api/mostsimilar/', methods=['POST'])
def most_similar():
    text = request.form['text']
    method = request.form['method']
    n = int(request.form['n'])
    titles, scores = d2v.infer(text, n, METHOD_DICT[method])
    return jsonify({'titles': titles, 'scores': scores})

if __name__ == '__main__':
    app.run(host='localhost', debug=False, threaded=True, use_reloader=False)