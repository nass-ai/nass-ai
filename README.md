# NASS-AI: Towards Digitization of Parliamentary Bills using Document Level Embedding and Bidirectional Long Short-Term Memory

Code for [https://arxiv.org/abs/1910.04865]()

A. Akinfaderin and O. Wahab. NASS-AI: Towards Digitization of Parliamentary Bills using Document Level Embedding and Bidirectional Long Short-Term Memory. To appear in the Proceedings of NeurIPS 2019 Workshop on Machine Learning for the Developing World, Vancouver, Canada, December 2019. 7 pages.

## Dataset Samples from Nigerian Parliamentary Bills
![Image description](https://s3.amazonaws.com/assertpub/image/1910.04865v1/image-002-000.png)



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
```python nass_ai preprocess --data [data_path]``` where: --data_path = Path to downloaded NASS crawl.
    
* Then, build word2vec and doc2vec embeddings.
        
    For word2vec:
    ``
    nassai.py build_embedding --cbow --epoch=[epoch choice]
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
