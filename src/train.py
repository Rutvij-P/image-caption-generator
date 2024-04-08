# train.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from data_preprocessing import load_coco_data, build_vocab_and_encode_captions, preprocess_image, simple_tokenizer

# Load the model parameters (vocab_size and max_length)
with open('models/model_params.json', 'r') as file:
    model_params = json.load(file)

vocab_size = model_params['vocab_size']
max_length = model_params['max_length']

# Import the model function
from models.caption_model import create_feature_extractor, create_caption_generator

# Loading dataset
image_paths, captions = load_coco_data('dataset')

# Train/validation split
train_image_paths, val_image_paths, train_captions, val_captions = train_test_split(
    image_paths,
    captions,
    test_size=0.2,  # 80% training, 20% validation
    random_state=42  # Seed for reproducibility
)

# Assume `build_vocab_and_encode_captions` is modified to return word_to_index and max_length as well
encoded_captions, word_to_index, index_to_word, _, _ = build_vocab_and_encode_captions(captions)

# Data generator
def data_generator(image_paths, captions, word_to_index, max_length, vocab_size, batch_size):
    X_images, X_seqs, y = [], [], []
    n = 0

    while True:
        for i in range(len(image_paths)):
            n += 1
            # Preprocess the image
            image = preprocess_image(image_paths[i])[0]  # Ensure this returns a single image array
            # Tokenize and encode the caption
            seq = [word_to_index[word] for word in simple_tokenizer(captions[i]) if word in word_to_index]
            
            # Split one sequence into multiple X, y pairs
            for j in range(1, len(seq)):  # Changed 'i' to 'j' to prevent shadowing the outer loop variable
                in_seq, out_seq = seq[:j], seq[j]
                in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post')[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                
                # Store
                X_images.append(image)
                X_seqs.append(in_seq)
                y.append(out_seq)
            
            # Yield batch data
            if n == batch_size:
                yield ([np.array(X_images), np.array(X_seqs)], np.array(y))  # Corrected format
                X_images, X_seqs, y = [], [], []
                n = 0

# Model creation
feature_extractor = create_feature_extractor()
caption_model = create_caption_generator(vocab_size, max_length, feature_extractor.output_shape[1])

# Compile the model
caption_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Prepare data generators
batch_size = 32
train_gen = data_generator(train_image_paths, train_captions, word_to_index, max_length, vocab_size, batch_size)
val_gen = data_generator(val_image_paths, val_captions, word_to_index, max_length, vocab_size, batch_size)

# Set up checkpointing
checkpoint = ModelCheckpoint('models/best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)

# Train the model
steps_per_epoch = len(train_image_paths) // batch_size
validation_steps = len(val_image_paths) // batch_size

caption_model.fit(
    x=train_gen,
    epochs=20,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    callbacks=[checkpoint],
    verbose=2
)
