import numpy as np
import pandas as pd
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from plot_utils import plot_confusion_matrix, plot_keywords

MAX_SEQUENCE_LENGTH = 1000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

class_to_predict = 'category'


def clean_sentence(sentence):
    tokenizer = RegexpTokenizer(r'\w+')
    sentence = tokenizer.tokenize(sentence)
    return [w.lower() for w in sentence]


class Cnn:
    def __init__(self):
        self.df = None
        self.labels = None
        self.text = None
        self.word_index = None
        self.all_classes = ["business", "entertainment", "politics", "sport", "tech"]

        # train and test data
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None

        # layer
        self.embedding_layer = None
        self.embeddings_index = None

        # tokenizer
        self.tokenizer = None

        self.model = None

    def read_data(self, file_name='input/bbc-text.csv'):
        """Read data from file and save it to pandas data frame"""

        df = pd.read_csv(file_name)
        df = df.dropna()
        df = df.reset_index(drop=True)
        print('Shape of dataset ', df.shape)
        print(df.columns)
        print('No. of unique classes', len(set(df[class_to_predict])))
        self.df = df
        # self.labels = df.columns

    def prepare_data(self):
        """Prepares data for processing"""
        # get list of all classes
        all_classes = sorted(set(self.df[class_to_predict]))

        # convert to dict to map them
        mapped_classes = dict((note, number) for number, note in enumerate(all_classes))

        # save mapped classes to data frame
        self.df[class_to_predict] = self.df[class_to_predict].apply(lambda i: mapped_classes[i])

        # print(self.df[class_to_predict])
        text = []
        labels = []

        for i in range(self.df.shape[0]):
            d = self.df['text'][i]
            text.append(d)
            labels.append(self.df[class_to_predict][i])

        self.labels = labels
        self.text = text

    def create_data(self):
        """Prepare train and test data"""
        data = []

        for sentence in self.text:
            tmp = self.sentence_to_input(sentence)
            data.append(tmp)

        data = pad_sequences(data, maxlen=MAX_SEQUENCE_LENGTH)

        labels = to_categorical(np.asarray(self.labels))
        print('Shape of Data Tensor:', data.shape)
        print('Shape of Label Tensor:', labels.shape)

        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]
        nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

        self.x_train = np.abs(data[:-nb_validation_samples])
        print('Shape of x train', self.x_train.shape)
        self.y_train = labels[:-nb_validation_samples]
        print('Shape of y train', self.y_train.shape)
        self.x_val = np.abs(data[-nb_validation_samples:])
        print('Shape of x val', self.x_val.shape)
        self.y_val = labels[-nb_validation_samples:]
        print('Shape of y val', self.y_val.shape)

    def create_embedding_layer_with_glove(self):
        """Create embedding layer"""
        sentence = ' '.join([str(x) for x in self.text])
        arr = clean_sentence(sentence)
        word_index = set(arr)
        print('Number of Unique Tokens', len(word_index))

        embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
        for i, word in enumerate(word_index):
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        self.embedding_layer = Embedding(len(word_index) + 1,
                                         EMBEDDING_DIM, weights=[embedding_matrix],
                                         input_length=MAX_SEQUENCE_LENGTH, trainable=True)

    def create_embeddings_index(self):
        """use glove to create word vectors and save it to file"""
        embeddings_index = {}
        f = open('input/glove.6B.100d.txt', encoding='utf8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Total %s word vectors in Glove 6B 100d.' % len(embeddings_index))

        # saving
        with open('calculated_models/embeddings_index.pickle', 'wb') as handle:
            pickle.dump(embeddings_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.embeddings_index = embeddings_index

    def sentence_to_input(self, sentence):
        """Map sentence to vector"""
        arr = clean_sentence(sentence)
        embedding_matrix = np.random.random((len(arr) + 1, EMBEDDING_DIM))

        for i, word in enumerate(arr):
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        return list(embedding_matrix.flatten())

    def train_cnn(self, file='model_cnn.hdf5'):
        """Train cnn"""
        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = self.embedding_layer(sequence_input)
        l_cov1 = Conv1D(128, 5, activation='relu')(embedded_sequences)
        l_pool1 = MaxPooling1D(5)(l_cov1)
        l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
        l_pool2 = MaxPooling1D(5)(l_cov2)
        l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)
        l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling
        l_flat = Flatten()(l_pool3)
        l_dense = Dense(128, activation='relu')(l_flat)
        preds = Dense(5, activation='softmax')(l_dense)

        model = Model(sequence_input, preds)
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])

        print("Simplified convolutional neural network")
        model.summary()
        cp = ModelCheckpoint(file, monitor='val_acc', verbose=1, save_best_only=True)

        history = model.fit(self.x_train,
                            self.y_train,
                            validation_data=(self.x_val, self.y_val),
                            epochs=15,
                            batch_size=16,
                            callbacks=[cp])

    def create_cnn_model(self, model_file_name):
        """
        This function creates cnn model and saves it to file
        default: calculated_models/model_cnn.hdf5.best4
        """
        self.read_data()
        self.prepare_data()

        self.create_embeddings_index()
        self.create_data()
        self.create_embedding_layer_with_glove()

        self.train_cnn('calculated_models/' + model_file_name)

    def predict(self, text, file='model_cnn.hdf5.best4'):
        """Open cnn model and make prediction"""
        self.model = self.model if self.model else load_model('calculated_models/' + file)

        # load embedding index
        if self.embeddings_index is None:
            with open('calculated_models/embeddings_index.pickle', 'rb') as handle:
                self.embeddings_index = pickle.load(handle)

        # create input data
        data = [self.sentence_to_input(text)]
        data = pad_sequences(data, maxlen=MAX_SEQUENCE_LENGTH)
        data = np.abs(data)

        # make prediction
        result = self.model.predict(data)

        # print(result)
        index = np.argmax(result)
        probability = result[0][index]

        # print("Predicted class %s with probability %.3f" % (self.all_classes[index], probability))

        return [self.all_classes[index], probability]

    def plot_confusion_matrix(self, y_test, y_predicted):
        cm = confusion_matrix(y_test, y_predicted)
        plot_confusion_matrix(cm, classes=self.all_classes, normalize=False)

    # def plot_keywords(self):
    #     if not self.keywords:
    #         keywords = self.get_keywords(n=10)
    #     else:
    #         keywords = self.keywords
    #     for genre in keywords:
    #         top = keywords[genre]['top']
    #         bottom = keywords[genre]['bottom']
    #         plot_keywords(top, bottom, self.classes_[genre])


def test(test_file_name, model_file_name, use_best=False):
    """test precalculated model"""
    true_scores = []
    pred_scores = []

    # init cnn
    _cnn = Cnn()

    # read test data set
    df = pd.read_csv(test_file_name, encoding='utf8')
    for index, row in df.iterrows():
        d = row['text']

        # get true score
        true_s = row["category"]
        true_scores.append(true_s)

        # make prediction
        if use_best:
            [pred, _] = _cnn.predict(d)
        else:
            [pred, _] = _cnn.predict(d, model_file_name)

        pred_s = pred
        pred_scores.append(pred_s)

    print("\t F1 (macro): %f" % f1_score(true_scores, pred_scores, average='macro'))
    print("\t F1 (micro): %f" % f1_score(true_scores, pred_scores, average='micro'))
    print("\t F1 (weighted): %f" % f1_score(true_scores, pred_scores, average='weighted'))

    cnn.plot_confusion_matrix(true_scores, pred_scores)


if __name__ == "__main__":
    # init
    cnn = Cnn()

    # file name to which save cnn model
    new_model_file_name = 'model_cnn.hdf5.test'

    # train model
    # cnn.create_cnn_model(new_model_file_name)

    # path to test file
    test_file = 'input/news_category_same_categories.csv'

    # test model
    use_best_model = True
    test(test_file, new_model_file_name, use_best_model)
