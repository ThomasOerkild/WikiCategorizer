import xml.etree.ElementTree as ET

class GenerateCorpus(object):
    def __init__(self, filenames, dictionary):
        self.filenames = filenames
        self.dictionary = dictionary

    def __iter__(self):
        for doc in self.filenames:
            with open(doc, 'r') as f:
                # The wiki files don't have a root, so it's not valid xml.
                # Therefore we enclose the document in a root tag
                doc_file = ET.fromstringlist(["<root>", f.read(), "</root>"])
                docs = [doc.text.split() for doc in doc_file]
                for doc in docs:
                    yield self.dictionary.doc2bow(doc.text.split())