# WikiCategorizer

## Quick start
Inference locally:
```
python most_similar.py -m /path/to/trained_doc2vec_Model -n 10 < testfile.txt
```

To run inference on a web service, start the web service:

```
python WebService.py
```

and then to run inference on the server:
```
python most_similar.py -u http://localhost:5000/api/mostsimilar/ < testfile.txt
```

## Group members:
* Thomas Ørkild - s154433@student.dtu.dk
* Christian Ingwersen - s154264@student.dtu.dk


## Problem description
Given a text, we want to use K-nearest neighbours to find similar wikipedia pages. To compute the KNN we will use an algorithm, like bag-of-words, n-grams or tfidf to convert wikipedia articles from text to feature vectors. The exact algortihm we will use, is to be decided, but it needs to produce vectors in a high-dimensional space, where similiar texts are located close to each other in this space.

To efficiently compute KNN on a big dataset like Wikipedia, we will implement it using MapReduce as described in the paper: [The k-Nearest Neighbor Algorithm Using MapReduce Paradigm](http://ijssst.info/Vol-15/No-3/data/3857a513.pdf).

We will create a command-line tool that, given a text, utilizes this technique to find to k most similiar Wikipedia articles, and use majority voting on the categories of the k articles, to find which category it should belong to.

To easily organize the huge Wikipedia dataset we store it in an SQL-database, that we query from.


## Time plan
The weeks refer to the semester weeks. 

- **Before official project start**
  - Before the project begins our plan is to download and investigate all of the data, so we are comfortable with the data before we have to work with it. In addition, we'll decide on an algorithm for converting the articles to high dimensional feature vectors.
  
- **Week 10**
  - Read and understand the paper [*The k-Nearest Neighbor Algorithm Using MapReduce Paradigm*](<http://ijssst.info/Vol-15/No-3/data/3857a513.pdf>)
  - Start implementing the KNN algorithm using MapReduce as described in the paper.
  
- **Week 11**
  - Improve the efficiency of the first basic implementation.
  - Finish and test the implementation.
  
- **Week 12**
  - Make a command-line tool utilizing the algorithm.
  - Document results and start writing the report.
  
- **Week 13**
  - Finalize project presentation and report. 
