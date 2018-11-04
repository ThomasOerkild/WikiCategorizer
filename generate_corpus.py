import re

class GenerateCorpus(object):
    def __init__(self, filenames, dictionary):
        self.filenames = filenames
        self.dictionary = dictionary

    def __iter__(self):
        for doc in self.filenames:
            with open(doc, 'r') as f:
                pages = re.split("<|>", f.read())
                for i in range(2, len(pages), 4):
                    yield self.dictionary.doc2bow(pages[i].split())