from tagged_document_generator import TaggedDocumentGenerator
import os
import glob
from gensim.models.doc2vec import Doc2Vec

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

home = os.path.expanduser("~")
path = os.path.join(home, "Documents", "text")
files = glob.glob(os.path.join(path, "**/wiki_*"), recursive=True)

trian_corpus = TaggedDocumentGenerator(files)
model = Doc2Vec(vector_size=100, min_count=5, workers=4)
model.build_vocab(trian_corpus, progress_per=10000)

print("Training...")
model.train(trian_corpus, total_examples=model.corpus_count, epochs=20, report_delay=10)
print("Done training.")

model.save(os.path.join(home, "Documents", "my_doc2vec_model_trained"))
