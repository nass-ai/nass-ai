# NASS-AI: Towards Digitization of Parliamentary Bills using Document Level Embedding and Bidirectional Long Short-Term Memory

Code for [https://arxiv.org/abs/1910.04865]()

A. Akinfaderin and O. Wahab. NASS-AI: Towards Digitization of Parliamentary Bills using Document Level Embedding and Bidirectional Long Short-Term Memory. To appear in the Proceedings of NeurIPS 2019 Workshop on Machine Learning for the Developing World, Vancouver, Canada, December 2019. 7 pages.

## Dataset Samples from Nigerian Parliamentary Bills
![Three different bills showing some of the challenging quality of our parliamentary bills. Left: a bill to regulate local government elections. Center: a bill to prohibit the use of life bullets or Nigerian army to quell civil protests. Right: a bill to provide free screening and treatment of cancer and brain tumor.](https://s3.amazonaws.com/assertpub/image/1910.04865v1/image-002-000.png)

### Link to Nigerian parliamentary bills from 1999-2018 in PDF format (pre-OCR)

Dataset: [https://drive.google.com/open?id=19bxftHKcAe8Lq_yH3w8xNMVI9gVInQp8]()

Raw data for parliamentary proceedings are available upon request.

### Dependencies
* Python3 (tested on 3.5, 3.6, 3.7)
* Install all dependencies: `pip install -r requirements.txt`
* Note that NLTK will need some extra steps if you've just installed it for the first time: 
```
Resource stopwords not found.
Please use te NLTK Downloader to obtain the resource:
>>> import nltk
>>> nltk.download('stopwords')
```

### Running on Colab
You can replicate the project (train and predict using our default settings) using the colab notebook here.


### Training your own models

* The models are trained on the [NASS]() data crawl (Last update was on [insert date])
	
* To start data preprocessing, go to the top-level directory and run:
```python nass_ai preprocess --data [data_path]``` where:
    
    --data_path = Path to downloaded NASS crawl.
    
* Then, build word2vec and doc2vec embeddings.
        
    For word2vec:
    ``
    nassai.py build_embedding --cbow=[0 for cbow 1 for skipgram] --size=[your embedding size] 
    --epoch=[epoch choice]
    ``

    For doc2vec:
    ``
    nassai.py build_embedding --dbow=[0 for dbow 1 for dm] --size=[your embedding size]     --epoch=[epoch choice]
    ``

* If you'd rather use glove as opposed to building your own embeddings, download GLOVE from [insert GLOVE link]() and unzip it into the models folder.

* You can then go ahead to train by running:
``python nass_ai train --epochs=[] --batch=[] --use_glove``

The above command runs data against all the available algorithms listed below.

You can also choose your algorithm by running:

``python nass_ai train --mode=[word2vec|doc2vec] --using=[sklearn|keras] --epochs=[] --batch=[] --use_glove=[0|1]``

## Credits/Authors

Adewale Akinfaderin and Olamilekan Wahab 
