# caption_model.py
from keras.models import Model, Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Add, Input
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import Adam
import json

# Create the CNN feature extractor
def create_feature_extractor():
    model = InceptionV3(weights='imagenet')
    model_new = Model(model.input, model.layers[-2].output)
    return model_new

# Create the RNN caption generator
def create_caption_generator(vocab_size, max_length, feature_size):
    # Feature extraction model
    inputs1 = Input(shape=(feature_size,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    # Sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    # Decoder model
    decoder1 = Add()([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    # Tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    
    return model

# Load vocab_size and max_length from the file
model_params_path = 'models/model_params.json'
with open(model_params_path, 'r') as f:
    model_params = json.load(f)

vocab_size = model_params['vocab_size']
max_length = model_params['max_length']

# Now use vocab_size and max_length for model creation
feature_extractor = create_feature_extractor()
caption_generator = create_caption_generator(vocab_size, max_length, feature_extractor.output_shape[-1])

# Compile and summarize the model
caption_generator.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
caption_generator.summary()
