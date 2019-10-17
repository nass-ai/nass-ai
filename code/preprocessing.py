import re
import string


import pandas
import nltk
from code.utils import get_path


def clean_text(text):
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = nltk.word_tokenize(text)
    text = " ".join(text)
    text = re.sub(r'\d+', '', text)
    return text


def preprocess_data(data_path):
    print("Starting processing ...")
    data = pandas.read_csv(data_path)
    nans = lambda data: data[data.isnull().any(axis=1)]
    data = data.drop(list(nans(data).index))
    data['clean_text'] = data['clean_text'].apply(clean_text)
    saved_data_path = get_path('data') + '/clean_data.csv'
    data[['clean_text', 'bill_class']].to_csv(saved_data_path, index=False)
    print("Preprocessing Complete")
    return data
