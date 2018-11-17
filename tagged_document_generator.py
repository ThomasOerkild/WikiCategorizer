import xml.etree.ElementTree as ET
from gensim.models.doc2vec import TaggedDocument

class TaggedDocumentGenerator(object):
    def __init__(self, filenames):
        self.filenames = filenames

    def __iter__(self):
        for doc in self.filenames:
            with open(doc, 'r') as f:
                # The wiki files don't have a root, so it's not valid xml.
                # Therefore we enclose the document in a root tag
                doc_file = ET.fromstringlist(["<root>", f.read(), "</root>"])
                for doc in doc_file:
                    yield TaggedDocument(words=doc.text.split(), 
                    tags=[doc.attrib['id']])