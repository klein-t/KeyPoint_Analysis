import gensim
import numpy as np
import gensim.downloader as gloader
import KPA
from KPA import KPA
import copy
import pandas as pd


class glove(KPA):
    def __init__(self, embedding_space_dimension = 300):
        super().__init__()

        self.emb_dim = embedding_space_dimension
        self.glove_model = gloader.load(f"glove-wiki-gigaword-{self.emb_dim}")
        self.vocab = {"id2word": {}, "word2id": {}, "vocabs":[]}

    # extract all sentences, build the set of tokens, assign to each an id
    def build_vocab(self):
        word_id = 0
        for index in [f"{type}_{dataset}" for dataset in self.datasets for type in self.types]:
            dataframe = self.dataframes[index]
            for column in dataframe.columns[1:3]:
                for sentence in dataframe[column]:
                    for word in sentence:
                        if word not in self.vocab['word2id'].keys():
                            self.vocab['word2id'][word] = word_id
                            self.vocab['id2word'][word_id] = word
                            word_id += 1
        self.vocab['vocabs'] = list(self.vocab['word2id'].keys())

    def build_embedding_matrix(self):
        embedding_matrix_shape = (len(self.vocab['vocabs']), self.emb_dim) 
        self.embedding_matrix = np.zeros(shape = embedding_matrix_shape, dtype = np.float32) 
        for word, idx in self.vocab['word2id'].items():
            try:
                embedded_word = self.glove_model[word]
            except(KeyError, TypeError):
                embedded_word = np.random.uniform(low = -0.1, high = 0.1, size = self.emb_dim)
            self.embedding_matrix[idx] = embedded_word

    def encoding(self):
        self.encoded_dataframes = copy.deepcopy(self.dataframes)
        for index in [f"{type}_{dataset}" for dataset in self.datasets for type in self.types]:
            dataframe = self.encoded_dataframes[index]
            for column in dataframe.columns[1:3]:
                dataframe[column] = dataframe[column].apply(lambda sentence: [self.vocab['word2id'][words] for words in sentence])

    def get_longest_sentence(self):
        self.max_len = 0
        for index in [f"arguments_{dataset}" for dataset in ['train', 'dev']]:
            dataframe = self.encoded_dataframes[index]
            for column in dataframe.columns[1:3]:
                    for sentence in dataframe[column]:
                        self.max_len = max(self.max_len, len(sentence))

    def alligment(self,id):
        if id in self.encoded_dataframes['arguments_test']['arg_id'].to_numpy():
            return self.encoded_dataframes['arguments_test'].loc[self.encoded_dataframes['arguments_test']['arg_id'] == id, 'argument'].iloc[0]
        elif id in self.encoded_dataframes['key_points_test']['key_point_id'].to_numpy():
            return self.encoded_dataframes['key_points_test'].loc[self.encoded_dataframes['key_points_test']['key_point_id'] == id, 'key_point'].iloc[0]

    def padding(self, lst):
        return lst + [0] * (self.max_len - len(lst))

    def processing(self):
        if not self.processed:
            super().processing()
        self.build_vocab()
        self.build_embedding_matrix()
        self.encoding()
        self.get_longest_sentence()

        self.alligned_dataframe = self.dataframes['labels_test'][['arg_id', 'key_point_id']].applymap(self.alligment)
        self.alligned_dataframe = self.alligned_dataframe.applymap(self.padding)

        # TODO for some reasons 'arg_id' and 'kp_id seem swapped in the alligned dataframe







if __name__ == "__main__":
    print('hello')
    g = glove(300)
    g.processing()


                



