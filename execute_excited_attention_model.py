import os
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Flatten, LayerNormalization, multiply, Multiply, Add, Concatenate, Reshape
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler
from scipy.io.arff import loadarff 
import scipy.stats
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import seaborn as sns
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

os.environ['PYTHONHASHSEED'] = '0'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

DIRECTORY = 'static/images/chat_images'

class Excitation(layers.Layer):
    def __init__(self, ratio=1, activation='elu', layer_name='ex', **kwargs):
        super(Excitation, self).__init__(**kwargs)
        self.ratio = ratio
        self.activation = activation
        self._name = layer_name

    def build(self, input_shape):
        channel_axis = 1 if tf.keras.backend.image_data_format() == "channels_first" else -1
        features = input_shape[channel_axis]
        self.dense1 = layers.Dense(int(features * self.ratio), kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', name=f'{self._name}_dense1')
        self.activation_layer = layers.Activation(self.activation)
        self.dense2 = layers.Dense(features, activation='softmax', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', name=f'{self._name}_dense2')
        super(Excitation, self).build(input_shape)

    def call(self, inputs):
        se_feature = tf.expand_dims(inputs, axis=1)
        se_feature = self.dense1(se_feature)
        se_feature = self.activation_layer(se_feature)
        se_feature = self.dense2(se_feature)
        if tf.keras.backend.image_data_format() == 'channels_first':
            se_feature = tf.keras.layers.Permute((3, 1))(se_feature)
        return se_feature

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(Excitation, self).get_config()
        config.update({'ratio': self.ratio, 'activation': self.activation, 'name': self._name})
        return config

class TrainableAttention(tf.keras.layers.Layer):
    """
    Trainable attention layer.
    """
    def _init_(self, **kwargs):
        super(TrainableAttention, self)._init_(**kwargs)

    def build(self, input_shape):
        self._A = tf.Variable(np.identity(input_shape[2])*0.0, trainable=True, name='trainableattentionweights', dtype=tf.float32)
        super(TrainableAttention, self).build(input_shape)

    def call(self, x):
        re = tf.keras.backend.dot(x, tf.keras.activations.sigmoid(self._A))
        return re

def get_tabular_model(num_features, num_classes=2):
    source = Input(shape=(num_features,), name='tabular')

    layers = []
    num_excitations = max(2,int(np.sqrt(num_features)))
    ar = np.linspace(1./num_excitations,1.0,num_excitations)
    for i in range(1,num_excitations+1):
        layers.append(Excitation(activation='elu', ratio = min(1,ar[i-1]), layer_name='ex_{}'.format(i))(source))
    x = Add()(layers)

    x = LayerNormalization(name='normalized_attention_1')(x)
    reshaped_source = Reshape((1, num_features))(source)
    x = multiply([reshaped_source, x], name='hadamard')  # Residual connection
    x = LayerNormalization(name='normalized_attention_2')(x)
    x = TrainableAttention(name='trainable_att')(x) #Trainable attention
    x = Dense(64, activation='elu')(x)
    x = Dense(32, activation='elu')(x)
    output = Flatten()(x)
    outputs = Dense(num_classes, activation='sigmoid')(output)

    return Model(source, outputs)

def convert_columns_to_float(data):
    for column in data.columns:
        # Check if the column is of datetime type
        if pd.api.types.is_datetime64_any_dtype(data[column]):
            # Convert datetime to numeric (timestamp), then to float
            data[column] = pd.to_numeric(data[column].astype('datetime64[ns]'))
        else:
            # Convert other types to float
            data[column] = data[column].astype(float)
    return data

def correlation_attention(weights, classes, columns):
    correlations = []
    for i in range(len(classes)):
        k = classes[i]
        corr_matrix, p_matrix = scipy.stats.spearmanr(weights[i], axis=0)
        correlations.append(corr_matrix)

    correlations = np.asarray(correlations, dtype=np.float32)
    mean_corr = np.mean(np.abs(correlations), axis=0)
    corr_matrix = mean_corr

    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(corr_matrix)] = True
    fig, ax = plt.subplots(figsize=(22, 10))
    ax = sns.heatmap(corr_matrix, mask=mask, annot=False, linewidths=0.1, fmt=".1f", cmap="viridis", xticklabels=columns, yticklabels=columns)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_title("Correlation Matrix of Excited Attention")
    plt.savefig(f'{DIRECTORY}/correlation_resumee.png')
    plt.close()
    return mean_corr


# Funzione che fa il crop del modello partendo dall'input fino a layer_name,
# quindi restituisce la X dopo aver eseguito il forward pass sul modello croppato
def instancewise_weight(X,y,model,layer_name='ex_1'):
    
    X_ = X

    # create a structure to more easily access layers by name
    layer_lookup = dict([(layer.name, layer) for layer in model.layers])
    layer_output = layer_lookup[layer_name].output
    # create a function that stops the forward pass at the layer we want
    partial_net = tf.keras.backend.function([model.input], [layer_output])

    #inp = X_[np.newaxis, ...]
    activation = partial_net([X_])[0] #cosi abbbiamo istanze * 1 * features
    activation = np.asarray(activation,dtype=np.float32)
    activation = activation.reshape(activation.shape[0],activation.shape[-1])
    #activation = activation[0, 0, :]
    return activation#  returns instance*features matrix

def plot_excited_attention_online(X, y, model, feature_names):
    layers_weights = {}
    layers_names = []
    col = feature_names
    excitations = 0

    sns.set(font_scale=0.9)

    for layer in model.layers:
        if 'ex_' in layer.name:
            excitations += 1
            layers_names.append(layer.name)

            attention = instancewise_weight(X, y, model, layer_name=layer.name)
            class_instances = {}
            for t in range(attention.shape[0]):
                if y[t] not in class_instances:
                    class_instances[y[t]] = []
                class_instances[y[t]].append(attention[t])

            keys = []
            layers_weights[layer.name] = {}
            for key in class_instances:
                keys.append(key)
                cw2 = np.sum(np.asarray(class_instances[key], dtype=np.float32), axis=0)
                cw2 = np.asarray(cw2).reshape(-1,)
                layers_weights[layer.name][key] = cw2

    fig, axs = plt.subplots(excitations, sharex=True, sharey=True)
    current = 0
    for layer_name in layers_weights:
        weights = []
        keys = []
        for classe in layers_weights[layer_name]:
            keys.append(classe)
            weights.append(layers_weights[layer_name][classe])
        axs[current].set_title('Excited Attention for ' + str(layer_name))
        sns.heatmap(weights, cmap="viridis", xticklabels=col, yticklabels=keys, ax=axs[current])
        current += 1
    plt.savefig(f'{DIRECTORY}/excited_attention.png')
    plt.close()

    normalized_att = instancewise_weight(X, y, model, layer_name='hadamard')
    class_instances = {}
    class_instances[0] = []
    for t in range(normalized_att.shape[0]):
        if y[t] not in class_instances:
            class_instances[y[t]] = []
        class_instances[y[t]].append(normalized_att[t])

    weights = []
    keys = []
    unhaltered_weights = []
    for key in class_instances:
        keys.append(key)
        cw2 = np.mean(np.abs(np.asarray(class_instances[key], dtype=np.float32)), axis=0)
        cw2 = np.asarray(cw2).reshape(-1,)
        weights.append(cw2)
        unhaltered_weights.append(np.asarray(class_instances[key], dtype=np.float32))

    sns.heatmap(weights, cmap="viridis", xticklabels=col, yticklabels=keys).set(title='Per Class Importances')
    plt.savefig(f'{DIRECTORY}/per_class_importances_resumee.png')
    plt.close()

    mean_corr = correlation_attention(unhaltered_weights, keys, col)
    return mean_corr

def get_sorted_correlations(mean_corr, feature_names):
    correlations = []
    
    for i in range(mean_corr.shape[0]):
        for j in range(i+1, mean_corr.shape[1]):
            correlations.append((feature_names[i], feature_names[j], mean_corr[i, j]))
    
    correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    return correlations

def get_important_features_and_correlated_features(dataframe, dep_var):
    
    print("\n\n---------------------\n\n")
    print("Dataframe shape: ", dataframe.shape)
    print("dep_var: ", dep_var)
    print("\n\n---------------------\n\n")

    batch_size = 1024
    start_lr = 0.001
    n_epochs = 200
    n_folds = 5
    patience = 50

    dataframe = dataframe.fillna(0)
    dataframe = dataframe.replace('?', 0)
    y = dataframe.iloc[:, -1]
    X = dataframe.iloc[:, :-1]
    X = convert_columns_to_float(X)
    feature_names = X.columns

    X = np.asarray(X)
    y = np.asarray(y)
    model = get_tabular_model(len(feature_names))
    print(model.summary())

    if isinstance(y[0], str):
        unique_labels = np.unique(y)
        label_mapping = {label: i for i, label in enumerate(unique_labels)}
        y = np.array([label_mapping[label] for label in y])
    y = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    num_classes = len(y_train[0])
    num_features = len(feature_names)
    
    model = get_tabular_model(num_features, num_classes)
    lr = tf.keras.optimizers.schedules.ExponentialDecay(start_lr, decay_steps=50, decay_rate=0.9, staircase=False)
    optimizer = tf.keras.optimizers.legacy.Adam(lr)
    model.compile(
        optimizer=optimizer, 
        loss='categorical_crossentropy', 
        metrics=['accuracy','AUC', tfa.metrics.F1Score(num_classes=num_classes, average='weighted')],
        run_eagerly=False
    )
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                        filepath=f'{DIRECTORY}/best_model.keras',
                        save_weights_only=False,
                        monitor='val_f1_score',#accuracy
                        mode='max',
                        save_best_only=True)
    model.fit(
        X_train, y_train, 
        epochs=n_epochs, batch_size=batch_size, 
        shuffle=True, validation_data=(X_test, y_test), 
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=patience, monitor='val_f1_score', mode='max'),model_checkpoint_callback]
    )
    model.load_weights(f'{DIRECTORY}/best_model.keras')

    # Create a new model that extracts the output of the Excitation layer
    excitation_output_model = Model(inputs=model.input,
                                    outputs=model.get_layer('hadamard').output)

    # Get the output of the Excitation layer for the test data
    excitation_outputs = excitation_output_model.predict(X_test)

    # Reshape the output to remove the batch dimension
    output_array = np.squeeze(excitation_outputs, axis=1)

    # Compute feature importances from the Excitation layer output
    importance_scores = np.mean(output_array, axis=0)
    top_50_indices = importance_scores.argsort()[-50:][::-1]
    top_50_scores = importance_scores[top_50_indices]
    top_50_features = [feature_names[i] for i in top_50_indices]

    top_5_features = []
    for i in range(5):
        feature_name = top_50_features[i]
        feature_score = top_50_scores[i]
        top_5_features.append((feature_name, feature_score))

    y_test_argmax = [np.argmax(y) for y in y_test]
    mean_corr = plot_excited_attention_online(X_test, y_test_argmax, model, feature_names)
    sorted_correlations = get_sorted_correlations(mean_corr, feature_names)
    result = {
        'top_5_features': top_5_features,
        'top_5_correlations': sorted_correlations[:5]
    }
    return result