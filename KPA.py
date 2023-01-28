import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize import wordpunct_tokenize


class KPA():
    def __init__(self):
        self.types = ['arguments', 'key_points', 'labels']
        self.datasets = ['train', 'dev', 'test']
        self.dataframes = {f"{type}_{dataset}": self.__download_files(type, dataset) for type in self.types for dataset in self.datasets}
    

    def __download_files(self, type, dataset):
        path = 'https://raw.githubusercontent.com/IBM/KPA_2021_shared_task/main/kpm_data/' + type + '_' + dataset + '.csv'
        data = pd.read_csv(filepath_or_buffer = path)
        return data


    def __preprocess(self, text):
        allowed_chars = re.compile('[^0-9a-z #+_]')
        text = re.sub(allowed_chars, '', text)
        return text.lower()


    def text_processing(self, dataframe):
        for column in ['argument','topic']:
            dataframe[column] = dataframe[column].apply(lambda sentence: self.__preprocess(sentence))


    def tokenize_dataset(self, dataframes):
        for index in [f"{type}_{dataset}" for dataset in self.datasets for type in ['argument','label']]:
            dataframe = dataframes[index]
            for column in dataframe.columns:
                dataframe[column] = dataframe[column].apply(wordpunct_tokenize)


if __name__ == "__main__":

    print('holaa')




