import numpy as np
import json
import os
import sys
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_preprocessing import load_coco_data, build_vocab_and_encode_captions, preprocess_image, simple_tokenizer
from models.caption_model import create_feature_extractor, create_caption_generator

# Load the model parameters (vocab_size and max_length)
with open('models/model_params.json', 'r') as file:
    model_params = json.load(file)

vocab_size = model_params['vocab_size']
max_length = model_params['max_length']

# Load dataset
image_paths, captions = load_coco_data('dataset')

# Preprocess images
processed_images = np.array([preprocess_image(img_path)[0] for img_path in image_paths])

# Build vocabulary and encode captions
encoded_captions, word_to_index, index_to_word, _, _ = build_vocab_and_encode_captions(captions)

# Prepare input-output pairs
X_seqs = []
y = []
for caption in captions:
    seq = [word_to_index[word] for word in simple_tokenizer(caption) if word in word_to_index]
    for i in range(1, len(seq)):
        in_seq, out_seq = seq[:i], seq[i]
        in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post')[0]
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
        X_seqs.append(in_seq)
        y.append(out_seq)

X_seqs = np.array(X_seqs)
y = np.array(y)
processed_images_expanded = np.repeat(processed_images, repeats=len(X_seqs) // len(processed_images), axis=0)

# Create a TensorFlow Dataset
dataset = tf.data.Dataset.from_tensor_slices((processed_images_expanded, X_seqs, y))
batch_size = 32
dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Model creation
feature_extractor = create_feature_extractor()
caption_model = create_caption_generator(vocab_size, max_length, feature_extractor.output_shape[1])

# Compile the model
caption_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set up checkpointing
checkpoint = ModelCheckpoint('models/best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)

# Train the model
steps_per_epoch = len(processed_images_expanded) // batch_size
caption_model.fit(
    dataset,
    epochs=20,
    steps_per_epoch=steps_per_epoch,
    callbacks=[checkpoint]
)
