import os
from generate_corpus import GenerateCorpus
import gensim
import re

# Wiki file path (Not full path just Documents/text etc)
wiki_path = "Documents/text"

# Output path for corpus and dictionary
out_path = "Documents/WikiCluster"

# Get all file names
home = os.path.expanduser("~")
path = os.path.join(home, wiki_path)
files = [file for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(path)]
         for file in sublist if not file.endswith(".DS_Store")]

# Limit amount of files during testing
files = files[:2]

# collect statistics about all tokens
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
