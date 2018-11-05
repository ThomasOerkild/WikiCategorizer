import os
from generate_corpus import GenerateCorpus
import gensim
import re
import multiprocessing as mp
import glob
import xml.etree.ElementTree as ET

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

def merge_file_with_dict(dictionary, file_path):
    with open(file_path, 'r') as f:
        # The wiki files don't have a root, so it's not valid xml.
        # Therefore we enclose the document in a root tag
        doc_file = ET.fromstringlist(["<root>", f.read(), "</root>"])
        docs = [doc.text.split() for doc in doc_file]
        tmp = gensim.corpora.Dictionary(docs)
        tmp.compactify()
        dictionary.merge_with(tmp)
    return dictionary

# Build dictionary
if parallel:
    dictionaries = [gensim.corpora.Dictionary()] * len(files)

    def build_dict(args):
        dictionary = args[0]
        doc = args[1]
        return merge_file_with_dict(dictionary, doc)

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
    for file in files: merge_file_with_dict(dictionary, file)

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
