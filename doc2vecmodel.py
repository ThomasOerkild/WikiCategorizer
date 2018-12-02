import nltk
from gensim.models.doc2vec import Doc2Vec
from urllib.request import urlopen
import json
from nltk.corpus import stopwords
import re
from scipy.spatial import cKDTree
from enum import Enum
import numpy as np
import glob
import os
from tagged_document_generator import TaggedDocumentGenerator
from gensim.similarities.index import AnnoyIndexer


class NNSMethod(Enum):
    BRUTE = 1
    KD_TREE = 2
    ANNOY = 3
    
    def __str__(self):
        return self.value


class Doc2VecModel:
    
    BASE_WIKI_QUERY = "https://en.wikipedia.org/w/api.php?action=query&format=json&pageids="
    stopword_list = stopwords.words('english')

    def __init__(self, modelname):
        self.modelname = modelname
        if self.modelname is None:
            from tagged_document_generator import TaggedDocumentGenerator
            import logging
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
            trian_corpus = self._get_training_iterator()
            self.model = Doc2Vec(vector_size=100, min_count=5, workers=7)
            self.model.build_vocab(trian_corpus, progress_per=10000)
        else:
            self.model = Doc2Vec.load(self.modelname)

        self.nns_method_init_dict = {NNSMethod.BRUTE: True,
                                    NNSMethod.KD_TREE: False,
                                    NNSMethod.ANNOY: False}

    def infer_file(self, filename, n=10):
        with open(filename, 'r') as f:
            lines = f.readlines()
        lines = ' '.join(lines)
        return self.infer(lines, n)

    def infer(self, string, n=10, nnsmethod=NNSMethod.ANNOY, annoymodelpath="gensim_annoy"):
        self._initialize_nns_method(nnsmethod, annoymodelpath)
        words = self._preprocess(string)
        # Set the random seed to make inferred vector determanistic
        self.model.random = np.random.mtrand.RandomState(1337)
        inferred_vector = self.model.infer_vector(words)
        ids, dists = self._calculate_most_similar(inferred_vector, n, nnsmethod)
        titles = self._get_title_from_pageids(ids)
        return titles, dists
    
    def train(self, epochs):
        import logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        trian_corpus = self._get_training_iterator()
        self.model.train(trian_corpus, total_examples=self.model.corpus_count, epochs=epochs, report_delay=10)
        self.model.save(self.modelname)

    def _initialize_nns_method(self, nnsmethod, annoymodelpath):
        if self.nns_method_init_dict[nnsmethod]: return
        if nnsmethod == NNSMethod.KD_TREE:
            print("Building KD tree..")
            self.tree = cKDTree(self.model.docvecs.vectors_docs)
            print("Finished building KD tree.")
            self.keys = list(self.model.docvecs.doctags.keys())
        elif nnsmethod == NNSMethod.ANNOY:
            self.annoy_indexer = AnnoyIndexer()
            self.annoy_indexer.load(annoymodelpath)
            self.annoy_indexer.model = self.model

    def _calculate_most_similar(self, vector, n, nnsmethod):
        if nnsmethod == NNSMethod.BRUTE:
            tops = self.model.docvecs.most_similar([vector], topn=n)
            return [t[0] for t in tops], [t[1] for t in tops]
        if nnsmethod == NNSMethod.KD_TREE:
            dists, indicies = self.tree.query(vector, k=n)
            return [self.keys[i] for i in indicies], dists
        if nnsmethod == NNSMethod.ANNOY:
            tops = self.model.docvecs.most_similar([vector], topn=n, indexer=self.annoy_indexer)
            return [t[0] for t in tops], [t[1] for t in tops]

    def _preprocess(self, string):
        string = string.lower()
        string = re.sub('[^a-z\s]+', '', string)
        words = nltk.word_tokenize(string)
        return [word for word in words if word not in self.stopword_list]

    def _get_training_iterator(self):
        home = os.path.expanduser("~")
        path = os.path.join(home, "Documents", "text") # Data is assumed to be in ~/Documents/text
        files = glob.glob(os.path.join(path, "**/wiki_*"), recursive=True)
        return TaggedDocumentGenerator(files)

    def _get_title_from_pageids(self, ids):
        ids = '|'.join(ids)
        query = self.BASE_WIKI_QUERY + ids
        response = urlopen(query)
        dic = json.loads(response.read())
        return [v['title']  if 'title' in v else "PageId: " + str(v['pageid']) for v in dic['query']['pages'].values()]
