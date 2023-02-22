import gensim
import numpy as np
import gensim.downloader as gloader
import KPA
from KPA import data
import copy
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import  classification_report, PrecisionRecallDisplay, roc_curve, balanced_accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, average_precision_score, PrecisionRecallDisplay, ConfusionMatrixDisplay
import os
from utils import unpack


class glove(data):
    def __init__(self, embedding_space_dimension = 300):
        super().__init__()

        self.emb_dim = embedding_space_dimension
        self.model_name = f"glove-wiki-gigaword-{self.emb_dim}"
        self.model_file = f"{self.model_name}.model"
        self.model_path = f"glove_model\{self.model_file}"
        
        if os.path.exists(self.model_path):
            self.glove_model = gensim.models.KeyedVectors.load(self.model_path)
        else:
            self.glove_model = gloader.load(self.model_name)
            self.glove_model.save(self.model_path)

        self.vocab = {"id2word": {}, "word2id": {}, "vocabs":[]}


    def build_embedding_matrix(self):
        embedding_matrix_shape = (len(self.vocab['vocabs']) + 1, self.emb_dim) 
        self.embedding_matrix = np.zeros(shape = embedding_matrix_shape, dtype = np.float32) 
        for word, idx in self.vocab['word2id'].items():
            try:
                embedded_word = self.glove_model[word]
            except(KeyError, TypeError):
                embedded_word = np.random.uniform(low = -0.1, high = 0.1, size = self.emb_dim)
            self.embedding_matrix[idx] = embedded_word

    def processing(self):
        if not self.processed:
            super().processing()
        self.build_embedding_matrix()


    def measure(self, set = 'test'):
        self.set = set
        self.scores = {}
        self.scores['sum'] = {}
        #self.scores['average'] = {}

        self.scores['list'] = []

        for _, row in self.alligned_dataframes[self.set].iterrows():
            encoded_arg = self.embedding_matrix[row['encoded_arg']]
            encoded_kp = self.embedding_matrix[row['encoded_kp']]

            #summing token embeddings to get sentence overall embedding, then calculating cosine similarity
            summed_arg = encoded_arg.sum(axis = 0).reshape(1, -1)
            summed_kp = encoded_kp.sum(axis = 0).reshape(1, -1)

            similarity_summed = cosine_similarity(summed_arg, summed_kp)

            self.scores['sum'][row['arg_id']] = {row['key_point_id'] : similarity_summed}
            self.scores['list'].append(similarity_summed)

        self.scores['list'] = np.asarray(unpack(self.scores['list']))
        print(f'we are evaluating similarity among arguments and key points in the {self.set} set')


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
    
    def run(self, set = 'test'):
        self.processing()
        self.measure(set)
        self.evaluate()


if __name__ == "__main__":
    g = glove()
    g.run()


                



