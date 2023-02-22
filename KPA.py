import numpy as np
import pandas as pd
import re
import copy
import nltk
from nltk.tokenize import wordpunct_tokenize
from utils import padding

 
class Data():
    def __init__(self):
        self.download_types = ['arguments', 'key_points', 'labels']
        self.types = self.download_types[:-1]
        self.datasets = ['train', 'dev', 'test']
        self.train_dev = self.datasets[:-1]

        self.raw_dataframes = {f"{type}_{dataset}": self.__download_files(type, dataset) for type in self.download_types for dataset in self.datasets}
        self.processed = False
        self.tokenized = False

        self.vocab = {"id2word": {}, "word2id": {}, "vocabs": []}


    def __download_files(self, type, dataset):
            if dataset == 'test':
                path = 'https://raw.githubusercontent.com/IBM/KPA_2021_shared_task/main/test_data/' + type + '_test.csv'
            else: 
                path = 'https://raw.githubusercontent.com/IBM/KPA_2021_shared_task/main/kpm_data/' + type + '_' + dataset + '.csv'
            data = pd.read_csv(filepath_or_buffer = path)
            return data


    def preprocess(self, text):
        allowed_chars = re.compile('[^a-zA-Z]')
        text = re.sub(allowed_chars, ' ', text)
        return text.lower()


    def text_processing(self):
        for index in [f"{type}_{dataset}" for dataset in self.datasets for type in self.types]:
            dataframe = self.raw_dataframes[index]
            for column in dataframe.columns[1:3]:
                dataframe[column] = dataframe[column].apply(self.preprocess)
        self.processed = True


    # in the other functions I pass A dataframe, here I pass ALL raw_dataframes, which approach is better and more intuitive? # TODO
    def tokenize(self):
        self.dataframes = copy.deepcopy(self.raw_dataframes)
        for index in [f"{type}_{dataset}" for dataset in self.datasets for type in self.types]:
            dataframe = self.dataframes[index]
            for column in dataframe.columns[1:3]:
                dataframe[column] = dataframe[column].apply(wordpunct_tokenize)
        self.tokenized = True

    def build_vocab(self):
        word_id = 1
        for index in [f"{type}_{dataset}" for dataset in self.datasets[:-1] for type in self.types]:
            dataframe = self.dataframes[index]
            for column in dataframe.columns[1:3]:
                for sentence in dataframe[column]:
                    for word in sentence:
                        if word not in self.vocab['word2id'].keys():
                            self.vocab['word2id'][word] = word_id
                            self.vocab['id2word'][word_id] = word
                            word_id += 1
        self.vocab['vocabs'] = list(self.vocab['word2id'].keys())

    def encoding(self):
        self.encoded_dataframes = copy.deepcopy(self.dataframes)
        for index in [f"{type}_{dataset}" for dataset in self.datasets for type in self.types]:
            dataframe = self.encoded_dataframes[index]
            for column in dataframe.columns[1:3]:
                dataframe[column] = dataframe[column].apply(
                    lambda sentence: [self.vocab['word2id'].get(word, 0) for word in sentence])


    def get_longest_sentence(self):
        self.max_len = 0
        for index in [f"arguments_{dataset}" for dataset in self.train_dev]:
            dataframe = self.encoded_dataframes[index]
            for column in dataframe.columns[1:3]:
                for sentence in dataframe[column]:
                    self.max_len = max(self.max_len, len(sentence))

    def alligment(self, id, dataset):
        if id in self.encoded_dataframes[f'arguments_{dataset}']['arg_id'].to_numpy():
            return self.encoded_dataframes[f'arguments_{dataset}'].loc[self.encoded_dataframes[f'arguments_{dataset}']['arg_id'] == id, 'argument'].iloc[0]
        elif id in self.encoded_dataframes[f'key_points_{dataset}']['key_point_id'].to_numpy():
            return self.encoded_dataframes[f'key_points_{dataset}'].loc[self.encoded_dataframes[f'key_points_{dataset}']['key_point_id'] == id, 'key_point'].iloc[0]


    def allign(self):
        self.alligned_dataframes = {}
        for dataset in self.datasets:
            self.alligned_dataframes[dataset] = self.dataframes[f'labels_{dataset}'][[
                'arg_id', 'key_point_id', 'label']]
            self.alligned_dataframes[dataset][['encoded_arg', 'encoded_kp']] = self.alligned_dataframes[dataset][[
                'arg_id', 'key_point_id']].applymap(lambda id: self.alligment(id, dataset))
            self.alligned_dataframes[dataset][['encoded_arg', 'encoded_kp']] = self.alligned_dataframes[dataset][[
                'encoded_arg', 'encoded_kp']].applymap(lambda lst: padding(lst, self.max_len))

# ['index'][['encoded_arg', 'encoded_kp']].applymap(lambda lst: padding(lst, self.max_len))

    def processing(self):
        self.text_processing()
        self.tokenize()
        self.build_vocab()
        self.encoding()
        self.get_longest_sentence()
        self.allign()
        self.processed = True


if __name__ == "__main__":
    print('hello main c:')