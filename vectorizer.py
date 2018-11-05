import os
from generate_corpus import GenerateCorpus
import gensim
import re
import multiprocessing as mp
import glob

# Parallel flag
parallel = True

# Wiki file path (Not full path just Documents/text etc)
wiki_path = os.path.join("Documents", "text")

# Output path for corpus and dictionary
out_path = os.path.join("Documents", "WikiCluster")

# Get all file names
home = os.path.expanduser("~")
path = os.path.join(home, wiki_path)
files = glob.glob(os.path.join(path, "**/wiki_*"), recursive=True)

# Limit amount of files during testing
files = files[:2]

# Build dictionary
if parallel:
    dictionaries = [gensim.corpora.Dictionary()] * len(files)


    def build_dict(args):
        dictionary = args[0]
        doc = args[1]
        print(doc)
        with open(doc, 'r') as f:
            pages = re.split("<|>", f.read())
            for i in range(2, len(pages), 4):
                tmp = gensim.corpora.Dictionary(page.split() for page in pages)
                tmp.compactify()
                dictionary.merge_with(tmp)
        return dictionary

    # Multiprocessing
    args = zip(dictionaries, files)
    pool = mp.Pool(processes=8)
    results = pool.map(build_dict, args)

    # Join results
    dictionary = gensim.corpora.Dictionary()
    for result in results:
        dictionary.merge_with(result)
else:
    dictionary = gensim.corpora.Dictionary()
    for doc in files:
        print(doc)
        with open(doc, 'r') as f:
            pages = re.split("<|>", f.read())
            for i in range(2, len(pages), 4):
                tmp = gensim.corpora.Dictionary(page.split() for page in pages)
                tmp.compactify()
                dictionary.merge_with(tmp)

# Save dictionary to file
dict_file = os.path.join(home, out_path, "wiki_dict.dict")
if not os.path.isfile(dict_file):
    dictionary.save(dict_file)

# Generate corpus without loading it into memory
corpus = GenerateCorpus(files, dictionary)

# Save corpus to file
corpus_file = os.path.join(home, out_path, "wiki_corpus.mm")
if not os.path.isfile(corpus_file):
    gensim.corpora.MmCorpus.serialize(corpus_file, corpus)
