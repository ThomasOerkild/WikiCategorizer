import argparse
import nltk
from gensim.models.doc2vec import Doc2Vec
from urllib.request import urlopen
import json

parser = argparse.ArgumentParser(description='Compute most similar wikipedia articles')
parser.add_argument('--filename', type=str, required=True, help="Name of file containing a text.")
parser.add_argument('--modelname', type=str, required=True, help="Name of saved model.")
parser.add_argument('-n', type=int, default=50, help="Number of similar documents to find.")

VECTOR_SIZE = 100
MIN_COUNT = 2
WORKERS = 4

BASE_WIKI_QUERY = "https://en.wikipedia.org/w/api.php?action=query&format=json&pageids="

def _load_model(modelname):
    model = Doc2Vec(vector_size=VECTOR_SIZE, min_count=MIN_COUNT, workers=WORKERS)
    model.load(modelname)
    return model

def _get_title_from_pageids(ids):
    ids = '|'.join(ids)
    query = BASE_WIKI_QUERY + ids
    response = urlopen(query)
    dic = json.loads(response.read())
    return [v['title'] for v in dic['query']['pages'].values()]


def _main(filename, modelname, n):
    with open(filename, 'r') as f:
        lines = f.readlines()
    lines = ' '.join(lines)
    words = nltk.word_tokenize(lines.lower())

    model = Doc2Vec.load(modelname)

    inferred_vector = model.infer_vector(words)
    sims = model.docvecs.most_similar([inferred_vector], topn=n)

    ids = [t[0] for t in sims]
    titles = _get_title_from_pageids(ids)

    print(f"-------- TOP {n} similar texts --------")
    print("TITLE \t SIMILARITY SCORE")
    for i in range(n):
        print(str(titles[i]) + "\t" + str(sims[i][1]))




if __name__ == "__main__":
    args = parser.parse_args()
    _main(args.filename, args.modelname, args.n)