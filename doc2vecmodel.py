import nltk
from gensim.models.doc2vec import Doc2Vec
from urllib.request import urlopen
import json
from nltk.corpus import stopwords
import re

class Doc2VecModel:

    BASE_WIKI_QUERY = "https://en.wikipedia.org/w/api.php?action=query&format=json&pageids="
    stopword_list = stopwords.words('english')

    def __init__(self, modelname):
        self.modelname = modelname
        if self.modelname is None:
            from tagged_document_generator import TaggedDocumentGenerator
            import logging
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
            home = os.path.expanduser("~")
            path = os.path.join(home, "Documents", "text") # Data is assumed to be in ~/Documents/text
            files = glob.glob(os.path.join(path, "**/wiki_*"), recursive=True)
            self.model = Doc2Vec(vector_size=200, min_count=5, workers=7)
            self.model.build_vocab(trian_corpus, progress_per=10000)
        else:
            self.model = Doc2Vec.load(self.modelname)

    def infer_file(self, filename, n=10):
        with open(filename, 'r') as f:
            lines = f.readlines()
        lines = ' '.join(lines)
        return self.infer(lines, n)

    def infer(self, string, n=10):
        words = self._preprocess(string)
        inferred_vector = self.model.infer_vector(words)
        tops = self.model.docvecs.most_similar([inferred_vector], topn=n)
        ids = [t[0] for t in tops]
        scores = [t[1] for t in tops]
        titles = self._get_title_from_pageids(ids)
        return titles, scores
    
    def train(self, epochs):
        import logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        self.model.train(trian_corpus, total_examples=self.model.corpus_count, epochs=epochs, report_delay=10)
        self.model.save(self.modelname)

    def _preprocess(self, string):
        string = string.lower()
        string = re.sub('[^a-z\s]+', '', string)
        words = nltk.word_tokenize(string)
        return [word for word in words if word not in self.stopword_list]

    def _get_title_from_pageids(self, ids):
        ids = '|'.join(ids)
        query = self.BASE_WIKI_QUERY + ids
        response = urlopen(query)
        dic = json.loads(response.read())
        return [v['title']  if 'title' in v else "PageId: " + str(v['pageid']) for v in dic['query']['pages'].values()]
