import gensim
import numpy as np
import gensim.downloader as gloader
from KPA import KPA
import copy


class glove(KPA):
    def __init__(self, embedding_space_dimension = 300):
        super().__init__()

        self.emb_dim = embedding_space_dimension
        self.glove_model = gloader.load(f"glove-wiki-gigaword-{self.emb_dim}")
        self.vocab = {"id2word": {}, "word2id": {}, "vocabs":[]}


    # extract all sentences, build the set of tokens, assign to each an id
    def buil_vocab(self):
        word_id = 0
        for index in [f"{type}_{dataset}" for dataset in self.datasets for type in ['arguments','key_points']]:
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
        for index in [f"{type}_{dataset}" for dataset in self.datasets for type in ['arguments','key_points']]:
            dataframe = self.encoded_dataframes[index]
            for column in dataframe.columns[1:3]:
                dataframe[column] = dataframe[column].apply(lambda sentence: [self.vocab['word2id'][words] for words in sentence])



    
if __name__ == "__main__":
    print('hello')
    #g = glove(300)
    #print(g.__dict__)


                



