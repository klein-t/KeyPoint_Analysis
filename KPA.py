import numpy as np
import pandas as pd
import re
import copy
import nltk
from nltk.tokenize import wordpunct_tokenize

 
class KPA():
    def __init__(self):
        self.download_types = ['arguments', 'key_points', 'labels']
        self.datasets = ['train', 'dev', 'test']
        self.raw_dataframes = {f"{type}_{dataset}": self.__download_files(type, dataset) for type in self.download_types for dataset in self.datasets}
        self.processed = False
        self.tokenized = False

    def __download_files(self, type, dataset):
            if dataset == 'test':
                path = 'https://raw.githubusercontent.com/IBM/KPA_2021_shared_task/main/test_data/' + type + '_test.csv'
            else: 
                path = 'https://raw.githubusercontent.com/IBM/KPA_2021_shared_task/main/kpm_data/' + type + '_' + dataset + '.csv'
            data = pd.read_csv(filepath_or_buffer = path)
            return data


    def __preprocess(self, text):
        allowed_chars = re.compile('[^0-9a-z #+_]')
        text = re.sub(allowed_chars, '', text)
        return text.lower()


    def __text_processing(self):
        for index in [f"{type}_{dataset}" for dataset in self.datasets for type in ['arguments','key_points']]:
            dataframe = self.raw_dataframes[index]
            for column in dataframe.columns[1:3]:
                dataframe[column] = dataframe[column].apply(self.__preprocess)
        self.processed = True

    # in the other functions I pass A dataframe, here I pass ALL raw_dataframes, which approach is better and more intuitive? # TODO
    def __tokenize(self):
        self.dataframes = copy.deepcopy(self.raw_dataframes)
        for index in [f"{type}_{dataset}" for dataset in self.datasets for type in ['arguments','key_points']]:
            dataframe = self.dataframes[index]
            for column in dataframe.columns[1:3]:
                dataframe[column] = dataframe[column].apply(wordpunct_tokenize)
        self.tokenized = True

    #it is just incredibly fast
    def processing(self):
        self.__text_processing()
        self.__tokenize()
        print('hello')


if __name__ == "__main__":
    print('hello')
    #instance = KPA()
    #instance.processing()
    #print(instance.raw_raw_dataframes.keys())
