# WikiCategorizer
## Group members:
* Thomas Ørkild - s154433@student.dtu.dk
* Christian Ingwersen - s154264@student.dtu.dk

The final project report can be downloadet from Dropbox [here.](https://www.dropbox.com/s/eb29nia0dav2dyj/Computational_Tools-final_report.pdf?dl=0)

## Problem description
In this project we have investigated different algorithms to find the K nearest Wikipedia articles to a given text. The idea with our tool is that Wikipedia contributers can write a new text and find all the similar articles already on Wikipedia. Our idea is that the contributers then can link these articles in the new text to help the reader find related knowledge. 

The tool have been implemented as an exam project in the DTU course [02807 Computational Tools for Data Science.](http://www2.compute.dtu.dk/courses/02807/)


## Steps to reproduce
1. Download the Wikipedia database from [here](https://archive.org/search.php?query=subject%3A%22enwiki%22%20AND%20subject%3A%22data%20dumps%22%20AND%20collection%3A%22wikimediadownloads%22).
2. Extract and clean the downloaded Wikipedia dump file, wiki-dump.bz2, using WikiExtractor
    ```
    python wikiextractor/WikiExtractor.py wiki-dump.bz2
    ```
3.  The model assumes the cleaned wiki files to be in `~/Documents/text`, so copy all the files to there. 
    ```
    mv text/ ~/Documents/
    ```
4. Train the gensim doc2vec model.
    ```
    python train_doc2vec.py
    ```
5. Start the webserver. The code assumes that the trained model is located at `~/Documents/my_doc2vec_model_trained`.
    ```
    python WebService.py
    ```

6. Use post-requests to query the web server. The first time this is run, it will generate the annoy index with 50 trees, which takes about 1,5 hour.
    ```
    python most_similar.py -u http://localhost:5000/api/mostsimilar/ --method annoy <<< "This is a test string"
    ```

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
 
