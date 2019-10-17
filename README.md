# NASS-AI: Towards Digitization of Parliamentary Bills using Document Level Embedding and Bidirectional Long Short-Term Memory

Code for [https://arxiv.org/abs/1910.04865]()

A. Akinfaderin and O. Wahab. NASS-AI: Towards Digitization of Parliamentary Bills using Document Level Embedding and Bidirectional Long Short-Term Memory. To appear in the Proceedings of NeurIPS 2019 Workshop on Machine Learning for the Developing World, Vancouver, Canada, December 2019. 7 pages.

## Dataset Samples from Nigerian Parliamentary Bills
![Image description](https://s3.amazonaws.com/assertpub/image/1910.04865v1/image-002-000.png)



### Dependencies
* Python3 (tested on 3.5, 3.6, 3.7)
* Install all dependencies: `pip install -r requirements.txt`
* Note that NLTK will need some extra hand-holding if you've installed it for the first time: 
```
Resource stopwords not found.
Please use te NLTK Downloader to obtain the resource:
>>> import nltk
>>> nltk.download('stopwords')
```

### Running on Colab
You can replicate the project (using our default settings) using the colab notebook here.


### Training your own models

* The models are trained on the [NASS]() data crawl (Last update was on)
	
* To start data preprocessing, go to the top-level directory and run:
```python nass_ai preprocess --data [data_path]``` where:
    
    --data_path = Path to downloaded NASS crawl.
    
* Then, build word2vec and doc2vec embeddings.
        
For word2vec:
``
nassai.py build_embedding --dbow=1
``
    
