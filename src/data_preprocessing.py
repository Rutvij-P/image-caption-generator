import numpy as np
import os
import json
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.sequence import pad_sequences as keras_pad_sequences

# Custom tokenizer function
def simple_tokenizer(text):
    # Split text by spaces for a simple tokenization
    return text.lower().split()

# Preprocess images
def preprocess_image(image_path, target_size=(299, 299)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Normalize the image
    return img_array

# Function to create a vocabulary and encode captions
def build_vocab_and_encode_captions(captions):
    # Build vocabulary
    vocab = set()
    for caption in captions:
        tokens = simple_tokenizer(caption)
        vocab.update(tokens)
    
    # Create word to index and index to word mappings
    word_to_index = {word: index for index, word in enumerate(vocab, start=1)}  # Start indexing from 1
    word_to_index['<pad>'] = 0  # Add a padding token
    index_to_word = {index: word for word, index in word_to_index.items()}
    
    # Encode the captions
    encoded_captions = []
    for caption in captions:
        tokens = simple_tokenizer(caption)
        encoded_caption = [word_to_index[word] for word in tokens if word in word_to_index]
        encoded_captions.append(encoded_caption)
    
    # Find the max length of the encoded captions
    max_length = max(len(caption) for caption in encoded_captions)

    # Return
    return encoded_captions, word_to_index, index_to_word, vocab, max_length

# Pad encoded sequences
def pad_encoded_captions(encoded_captions, max_length):
    return keras_pad_sequences(encoded_captions, maxlen=max_length, padding='post')


# Load and preprocess COCO data
def load_coco_data(base_dir='dataset', annotation_file='annotations/captions_train2017.json'):
    annotation_path = os.path.join(base_dir, annotation_file)
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)

    image_paths = []
    captions = []
    for annot in annotations['annotations']:
        image_id = annot['image_id']
        image_path = os.path.join(base_dir, 'images', f'{str(image_id).zfill(12)}.jpg')
        
        # Check if the image file exists before adding it to the list
        if os.path.exists(image_path):
            image_paths.append(image_path)
            caption = f"<start> {annot['caption']} <end>"
            captions.append(caption)
        else:
            print(f"Skipping missing image: {image_path}")

    return image_paths, captions