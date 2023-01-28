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
            dataframe = self.dataframes[index]
            for column in dataframe.columns[1:3]:
                dataframe[column] = dataframe[column].apply(lambda sentence: self.__preprocess(sentence))

    # in the other functions I pass A dataframe, here I pass ALL dataframes, which approach is better and more intuitive? # TODO
    def __tokenize(self):
        for index in [f"{type}_{dataset}" for dataset in self.datasets for type in ['arguments','key_points']]:
            dataframe = self.dataframes[index]
            for column in dataframe.columns[1:3]:
                dataframe[column] = dataframe[column].apply(wordpunct_tokenize)

    def processing(self):
        self.__text_processing()
        self.__tokenize()


if __name__ == "__main__":

    instance = KPA()
    instance.processing()
    print(instance.dataframes.keys())




