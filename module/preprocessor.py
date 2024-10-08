import io
import re
import string
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class Preprocessor(object):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.classes = self.config['classes']
        self._load_data()

    @staticmethod
    def clean_text(text):
        text = text.strip().lower().replace('\n', '')
        # tokenization
        words = re.split(r'\W+', text)  # or just words = text.split()
        # filter punctuation
        filter_table = str.maketrans('', '', string.punctuation)
        clean_words = [w.translate(filter_table) for w in words if len(w.translate(filter_table))]
        return clean_words

    def _parse(self, data_frame, is_test = False):
        """
            input: 
                data_frame
            output:
                tokenized_input (np.array) # [i, haven, t, paraphrased, you, at, all, gary,...]
                n_hot_label (np.array) # [0, 1, 0, 0, 1]
                    or
                test_id if is_test == True
        """
        X = data_frame[self.config['input_text_column']].apply(Preprocessor.clean_text).values

        Y = None
        if is_test == False:
            Y = data_frame.drop([self.config['input_id_column'], self.config['input_text_column']], axis=1).values
        else:
            # Y = data_frame.id.values
            Y = data_frame[self.config['input_id_column']].values

        return X, Y

    def _load_data(self): 
        data_df = pd.read_csv(self.config['input_trainset']) 
        data_df[self.config['input_text_column']].fillna("unknown", inplace=True)
        self.data_x, self.data_y = self._parse(data_df) 
        self.train_x,self.validate_x, self.train_y, self.validate_y = \
        train_test_split(self.data_x, self.data_y, test_size=self.config['split_ratio'], random_state=self.config['random_seed'])

        test_df = pd.read_csv(self.config['input_testset']) 
        test_df[self.config['input_text_column']].fillna("unknown", inplace=True)
        self.test_x, self.test_ids = self._parse(test_df, is_test=True) 

    def process(self):
        input_convertor = self.config.get('input_convertor', None)
        data_x, data_y, train_x, train_y, validate_x, validate_y, test_x = \
                self.data_x, self.data_y, self.train_x, self.train_y, \
                self.validate_x, self.validate_y, self.test_x

        if input_convertor == 'count_vectorization':
            data_x, train_x, validate_x, test_x = self.count_vectorization(data_x, train_x, validate_x, test_x)
        elif input_convertor == 'tfidf_vectorization':
            data_x, train_x, validate_x, test_x= self.tfidf_vectorization(data_x, train_x, validate_x, test_x)
        elif input_convertor == 'nn_vectorization': # for neural network
            data_x, train_x, validate_x, test_x = self.nn_vectorization(data_x, train_x, validate_x, test_x)

        return data_x, data_y, train_x, train_y, validate_x, validate_y, test_x

    def count_vectorization(self, data_x, train_x, validate_x, test_x):
        vectorizer = CountVectorizer(tokenizer=lambda x:x, preprocessor=lambda x:x)
        vectorized_data_x = vectorizer.fit_transform(data_x)
        # Print vocabulary
        # vocabulary = vectorizer.get_feature_names_out()
        # print("vocabulary:", vocabulary)
        vectorized_train_x  = vectorizer.transform(train_x)
        vectorized_validate_x  = vectorizer.transform(validate_x)
        vectorized_test_x  = vectorizer.transform(test_x)
        return vectorized_data_x, vectorized_train_x, vectorized_validate_x, vectorized_test_x

    def tfidf_vectorization(self, data_x, train_x, validate_x, test_x):
        vectorizer = TfidfVectorizer(tokenizer=lambda x:x, preprocessor=lambda x:x)
        vectorized_data_x = vectorizer.fit_transform(data_x)
        # Print vocabulary
        # vocabulary = vectorizer.get_feature_names_out()
        # print("vocabulary:", vocabulary)
        vectorized_train_x = vectorizer.transform(train_x)
        vectorized_validate_x  = vectorizer.transform(validate_x)
        vectorized_test_x  = vectorizer.transform(test_x)
        return vectorized_data_x, vectorized_train_x, vectorized_validate_x, vectorized_test_x

    def nn_vectorization(self, data_x, train_x, validate_x, test_x):
        self.word2ind = {}
        self.ind2word = {}

        specialtokens = ['<pad>', '<unk>']

        pretrained_embedding = self.config.get('pretrained_embedding', None)

        if pretrained_embedding is not None:
            word2embedding = Preprocessor.load_vectors(pretrained_embedding)
            vocabs = specialtokens + list(word2embedding.keys())
            vocabs_size = len(vocabs)
            self.embedding_matrix = np.zeros((vocabs_size, self.config['embedding_dim']))
            for token in specialtokens:
                word2embedding[token] = np.random.uniform(low=-1, high=1, size=(self.config['embedding_dim'],))
            for idx, word in enumerate(vocabs):
                self.word2ind[word] = idx
                self.ind2word[idx] = word
                self.embedding_matrix[idx] = word2embedding[word]
        else:
            def addword(word2ind, ind2word, word):
                if word in word2ind:
                    return
                ind2word[len(word2ind)] = word
                word2ind[word] = len(word2ind)

            for token in specialtokens:
                addword(self.word2ind, self.ind2word, token)

            for sent in data_x:
                for word in sent:
                    addword(self.word2ind, self.ind2word, word)

        def tokenize_sentences(data):
            x_ids = []
            for sent in data:
                indsent = [self.word2ind.get(word, self.word2ind['<unk>']) for word in sent]
                x_ids.append(indsent)
            x_ids = keras.preprocessing.sequence.pad_sequences(x_ids, maxlen=self.config['maxlen'], padding='post',value=self.word2ind['<pad>'])
            x_ids = np.array(x_ids)
            return x_ids

        data_x_ids = tokenize_sentences(data_x)
        train_x_ids = tokenize_sentences(train_x)
        validate_x_ids = tokenize_sentences(validate_x)
        test_x_ids = tokenize_sentences(test_x)

        return data_x_ids, train_x_ids, validate_x_ids, test_x_ids

    @staticmethod
    def load_vectors(fname):
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        #n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = np.array(list(map(float, tokens[1:])))
        return data
