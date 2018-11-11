from generate_corpus import GenerateTaggedDocuments
import os
import glob
from gensim.models.doc2vec import Doc2Vec

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

home = os.path.expanduser("~")
path = os.path.join(home, "Documents", "text")
files = glob.glob(os.path.join(path, "**/wiki_*"), recursive=True)

trian_corpus = GenerateTaggedDocuments(files)
model = Doc2Vec(vector_size=100, min_count=2, workers=4)
model.build_vocab(trian_corpus, progress_per=10000)

model.save("my_doc2vec_model_untrained")

print("Training...")
model.train(trian_corpus, total_examples=model.corpus_count, epochs=model.epochs, report_delay=10)
print("Done training.")

#inferred_vector = model.infer_vector(test_string)
#sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

model.save("my_doc2vec_model_trained")
