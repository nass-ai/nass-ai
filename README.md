# NASS-AI: Towards Digitization of Parliamentary Bills using Document Level Embedding and Bidirectional Long Short-Term Memory

Code for [https://arxiv.org/abs/1910.04865]()

A. Akinfaderin and O. Wahab. NASS-AI: Towards Digitization of Parliamentary Bills using Document Level Embedding and Bidirectional Long Short-Term Memory. To appear in the Proceedings of NeurIPS 2019 Workshop on Machine Learning for the Developing World, Vancouver, Canada, December 2019. 7 pages.

## Dataset Samples from Nigerian Parliamentary Bills




### Dependencies
* Python3 (tested on 3.5, 3.6, 3.7)
* Install all dependencies: `pip3 install -r requirements.txt`

### Reproduction Option #1
Run this notebook [Add notebook link here]()

### Reproduction Option #2

1) From the top-level directory, run `python nass_ai predict --data bill.txt --dbow=1` where :

    ```
    --data = Path of bill to predict
    
    --dbow = Predict using DBOW doc2vec model or DM doc2vec model
    ```

### Train your own model

* Your data should be a csv with 2 columns bill_text and bill_class

* Install dependencies: `pip3 install -r requirements.txt`
* Note that NLTK will need some extra hand-holding if you've installed it for the first time: 
	```
	Resource stopwords not found.
  	Please use the NLTK Downloader to obtain the resource:

  	>>> import nltk
  	>>> nltk.download('stopwords')
	```
	
To start data-prep, go to the top-level directory and run:
    `python nass_ai preprocess --data [data_path.txt]` where:
    
    --data_path.txt = Path of bill to predict
    
    --dbow = Predict using DBOW doc2vec model or DM doc2vec model
    
You can then build your doc2vec model by running:
``
nassai.py train_doc2vec --dbow=1
``
    
