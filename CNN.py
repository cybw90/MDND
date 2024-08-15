from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load the complete dataset
data = pd.read_csv("UCIF.csv")
print("Original Data Shape:", data.shape)


data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(subset=['URLs', 'Label'], inplace=True)  # Only essential columns

# Normalize/standardize features
scaler = StandardScaler()
features = data.drop(['URLs', 'Label'], axis=1).select_dtypes(include=[np.number])
features_scaled = scaler.fit_transform(features)

# Tokenize and pad URL data for CNN
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(data['URLs'])
sequences = tokenizer.texts_to_sequences(data['URLs'])
urls_padded = pad_sequences(sequences, maxlen=100)

# Prepare labels
labels = data['Label'].values
labels_encoded = to_categorical(labels)

# Define the CNN model
def create_cnn_model(vocabulary_size, embedding_dim=50):
    model = Sequential([
        Embedding(input_dim=vocabulary_size, output_dim=embedding_dim),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.5),
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.5),
        Dense(len(np.unique(labels)), activation='softmax')
    ])
    return model

# Prepare for cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1

for train_index, test_index in skf.split(urls_padded, labels):
    urls_train, urls_test = urls_padded[train_index], urls_padded[test_index]
    y_train, y_test = labels_encoded[train_index], labels_encoded[test_index]
    
    # Build and compile the model
    cnn_model = create_cnn_model(len(tokenizer.word_index) + 1)
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    print(f'Training for fold {fold_no} ...')
    history = cnn_model.fit(urls_train, y_train, epochs=10, batch_size=32, validation_data=(urls_test, y_test))
    
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Model accuracy for Fold {fold_no}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model Loss for Fold {fold_no}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Increase fold number
    fold_no += 1
