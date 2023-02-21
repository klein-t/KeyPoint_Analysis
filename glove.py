import gensim
import numpy as np
import gensim.downloader as gloader
import KPA
from KPA import KPA
import copy
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import  classification_report, PrecisionRecallDisplay, roc_curve, balanced_accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, average_precision_score, PrecisionRecallDisplay, ConfusionMatrixDisplay
import os



class glove(KPA):
    def __init__(self, set = 'test', embedding_space_dimension = 300):
        super().__init__()

        self.emb_dim = embedding_space_dimension
        self.set = set
        self.model_name = f"glove-wiki-gigaword-{self.emb_dim}"
        self.model_file = f"{self.model_name}.model"
        self.model_path = f"glove_model\{self.model_file}"
        
        if os.path.exists(self.model_path):
            self.glove_model = gensim.models.KeyedVectors.load(self.model_path)
        else:
            self.glove_model = gloader.load(self.model_name)
            self.glove_model.save(self.model_path)

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
        if id in self.encoded_dataframes[f'arguments_{self.set}']['arg_id'].to_numpy():
            return self.encoded_dataframes[f'arguments_{self.set}'].loc[self.encoded_dataframes[f'arguments_{self.set}']['arg_id'] == id, 'argument'].iloc[0]
        elif id in self.encoded_dataframes[f'key_points_{self.set}']['key_point_id'].to_numpy():
            return self.encoded_dataframes[f'key_points_{self.set}'].loc[self.encoded_dataframes[f'key_points_{self.set}']['key_point_id'] == id, 'key_point'].iloc[0]


    def unpack(self, data):
        for i, arr in enumerate(data):
            data[i] = arr[0][0]
        return data

    def padding(self, lst):
        return lst + [0] * (self.max_len - len(lst))


    def processing(self):
        if not self.processed:
            super().processing()
        self.build_vocab()
        self.build_embedding_matrix()
        self.encoding()
        self.get_longest_sentence()
        self.alligned_dataframe = self.dataframes[f'labels_{self.set}'][['arg_id', 'key_point_id']]
        self.alligned_dataframe[['encoded_arg_id', 'encoded_key_point_id']] = self.dataframes[f'labels_{self.set}'][['arg_id', 'key_point_id']].applymap(self.alligment)
        self.alligned_dataframe[['encoded_arg_id', 'encoded_key_point_id']] = self.alligned_dataframe[['encoded_arg_id', 'encoded_key_point_id']].applymap(self.padding)

    def measure(self):
        self.scores = {}
        self.scores['sum'] = {}
        #self.scores['average'] = {}

        self.scores['list'] = []

        for _, row in self.alligned_dataframe.iterrows():
            encoded_arg = self.embedding_matrix[row['encoded_arg_id']]
            encoded_kp = self.embedding_matrix[row['encoded_key_point_id']]

            #summing token embeddings to get sentence overall embedding, then calculating cosine similarity
            summed_arg = encoded_arg.sum(axis = 0).reshape(1, -1)
            summed_kp = encoded_kp.sum(axis = 0).reshape(1, -1)

            similarity_summed = cosine_similarity(summed_arg, summed_kp)

            self.scores['sum'][row['arg_id']] = {row['key_point_id'] : similarity_summed}
            self.scores['list'].append(similarity_summed)

        self.scores['list'] = np.asarray(self.unpack(self.scores['list']))


    def evaluate(self): 
            predictions = self.scores['list']
            true_labels = self.dataframes[f'labels_{self.set}']['label'].to_list()
            fpr, tpr, thresholds = roc_curve(true_labels, predictions)
            thr = thresholds[np.argmin(np.abs(fpr + tpr - 1))]

            predictions_thr = np.zeros(predictions.shape)
            predictions_thr[predictions >= thr] = 1

            print(classification_report(predictions_thr, true_labels))
                
            print(f'The average precision score is: {average_precision_score(true_labels, predictions_thr)}.')
            print(f'The balanced accuracy score is: {balanced_accuracy_score(true_labels, predictions_thr)}.')
            print(f'The tuned threshold is: {thr}.')

            print(f'The confusion matrix is: ')
            ConfusionMatrixDisplay.from_predictions(true_labels, predictions_thr)

            PrecisionRecallDisplay.from_predictions(true_labels, predictions)
    
    def run(self):
        self.processing()
        self.measure()
        self.evaluate()


if __name__ == "__main__":
    g = glove()
    g.run()


                



