CMU_DICT_PATH = 'C:\\Users\\ChuuChuuPC\\hosting\\model\\cmudict-0 (1).7b'
CMU_SYMBOLS_PATH = 'C:\\Users\\ChuuChuuPC\\hosting\\model\\cmudict.symbols'
import re
import random
import numpy as np

IS_KAGGLE = True

ILLEGAL_CHAR_REGEX = "[^A-Z-'.]"
MAX_DICT_WORD_LEN = 20
MIN_DICT_WORD_LEN = 1  # Corrected indentation

def load_clean_phonetic_dictionary():
    def is_alternate_pho_spelling(word):
        return '(' in word and ')' in word

    def should_skip(word):
        if not word[0].isalpha():
            return True
        if word[-1] == '.':
            return True
        if re.search(ILLEGAL_CHAR_REGEX, word):
            return True
        if len(word) > MAX_DICT_WORD_LEN:
            return True
        if len(word) < MIN_DICT_WORD_LEN:
            return True
        return False

    phonetic_dict = {}
    with open(CMU_DICT_PATH, encoding="ISO-8859-1") as cmu_dict:
        for line in cmu_dict:
            if line[0:3] == ';;;':
                continue

            word, phonetic = line.strip().split('  ')

            if is_alternate_pho_spelling(word):
                word = word[:word.find('(')]

            if should_skip(word):
                continue

            if word not in phonetic_dict:
                phonetic_dict[word] = []
            phonetic_dict[word].append(phonetic)

        if IS_KAGGLE:
            phonetic_dict = {key: phonetic_dict[key] for key in random.sample(list(phonetic_dict.keys()), 5000)}

    return phonetic_dict

phonetic_dict = load_clean_phonetic_dictionary()
example_count = np.sum([len(prons) for _, prons in phonetic_dict.items()])


print("\n".join([k+' --> '+phonetic_dict[k][0] for k in random.sample(list(phonetic_dict.keys()), 10)]))
print('\nAfter cleaning, the dictionary contains %s words and %s pronunciations (%s are alternate pronunciations).' %
      (len(phonetic_dict), example_count, (example_count-len(phonetic_dict))))

import string

START_PHONE_SYM = '\t'
END_PHONE_SYM = '\n'


def char_list():
    allowed_symbols = [".", "-", "'"]
    uppercase_letters = list(string.ascii_uppercase)
    return [''] + allowed_symbols + uppercase_letters


def phone_list():
    phone_list = [START_PHONE_SYM, END_PHONE_SYM]
    with open(CMU_SYMBOLS_PATH) as file:
        for line in file:
            phone_list.append(line.strip())
    return [''] + phone_list


def id_mappings_from_list(str_list):
    str_to_id = {s: i for i, s in enumerate(str_list)}
    id_to_str = {i: s for i, s in enumerate(str_list)}
    return str_to_id, id_to_str


# Create character to ID mappings
char_to_id, id_to_char = id_mappings_from_list(char_list())

# Load phonetic symbols and create ID mappings
phone_to_id, id_to_phone = id_mappings_from_list(phone_list())

# Example:
#print('Char to id mapping: \n', char_to_id)


CHAR_TOKEN_COUNT = len(char_to_id)
PHONE_TOKEN_COUNT = len(phone_to_id)


def char_to_1_hot(char):
    char_id = char_to_id[char]
    hot_vec = np.zeros((CHAR_TOKEN_COUNT))
    hot_vec[char_id] = 1.
    return hot_vec


def phone_to_1_hot(phone):
    phone_id = phone_to_id[phone]
    hot_vec = np.zeros((PHONE_TOKEN_COUNT))
    hot_vec[phone_id] = 1.
    return hot_vec

# Example:
#print('"A" is represented by:\n', char_to_1_hot('A'), '\n-----')
#print('"AH0" is represented by:\n', phone_to_1_hot('AH0'))

MAX_CHAR_SEQ_LEN = max([len(word) for word, _ in phonetic_dict.items()])
MAX_PHONE_SEQ_LEN = max([max([len(pron.split()) for pron in pronuns])
                         for _, pronuns in phonetic_dict.items()]
                       ) + 2  # + 2 to account for the start & end tokens we need to add


def dataset_to_1_hot_tensors():
    char_seqs = []
    phone_seqs = []

    for word, pronuns in phonetic_dict.items():
        word_matrix = np.zeros((MAX_CHAR_SEQ_LEN, CHAR_TOKEN_COUNT))
        for t, char in enumerate(word):
            word_matrix[t, :] = char_to_1_hot(char)
        for pronun in pronuns:
            pronun_matrix = np.zeros((MAX_PHONE_SEQ_LEN, PHONE_TOKEN_COUNT))
            phones = [START_PHONE_SYM] + pronun.split() + [END_PHONE_SYM]
            for t, phone in enumerate(phones):
                pronun_matrix[t,:] = phone_to_1_hot(phone)

            char_seqs.append(word_matrix)
            phone_seqs.append(pronun_matrix)

    return np.array(char_seqs), np.array(phone_seqs)


char_seq_matrix, phone_seq_matrix = dataset_to_1_hot_tensors()
#print('Word Matrix Shape: ', char_seq_matrix.shape)
#print('Pronunciation Matrix Shape: ', phone_seq_matrix.shape)

phone_seq_matrix_decoder_output = np.pad(phone_seq_matrix,((0,0),(0,1),(0,0)), mode='constant')[:,1:,:]

from keras.models import Model
from keras.layers import Input, LSTM, Dense

def baseline_model(hidden_nodes = 256):

    # Shared Components - Encoder
    global testing_encoder_model,testing_decoder_model
    char_inputs = Input(shape=(None, CHAR_TOKEN_COUNT))
    encoder = LSTM(hidden_nodes, return_state=True)

    # Shared Components - Decoder
    phone_inputs = Input(shape=(None, PHONE_TOKEN_COUNT))
    decoder = LSTM(hidden_nodes, return_sequences=True, return_state=True)
    decoder_dense = Dense(PHONE_TOKEN_COUNT, activation='softmax')

    # Training Model
    _, state_h, state_c = encoder(char_inputs) # notice encoder outputs are ignored
    encoder_states = [state_h, state_c]
    decoder_outputs, _, _ = decoder(phone_inputs, initial_state=encoder_states)
    phone_prediction = decoder_dense(decoder_outputs)

    training_model = Model([char_inputs, phone_inputs], phone_prediction)

    # Testing Model - Encoder
    testing_encoder_model = Model(char_inputs, encoder_states)

    # Testing Model - Decoder
    decoder_state_input_h = Input(shape=(hidden_nodes,))
    decoder_state_input_c = Input(shape=(hidden_nodes,))
    decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, decoder_state_h, decoder_state_c = decoder(phone_inputs, initial_state=decoder_state_inputs)
    decoder_states = [decoder_state_h, decoder_state_c]
    phone_prediction = decoder_dense(decoder_outputs)

    testing_decoder_model = Model([phone_inputs] + decoder_state_inputs, [phone_prediction] + decoder_states)

    return training_model, testing_encoder_model, testing_decoder_model

from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2

(char_input_train, char_input_test,
 phone_input_train, phone_input_test,
 phone_output_train, phone_output_test) = train_test_split(
    char_seq_matrix, phone_seq_matrix, phone_seq_matrix_decoder_output,
    test_size=TEST_SIZE, random_state=42)

TEST_EXAMPLE_COUNT = char_input_test.shape[0]

from keras.callbacks import ModelCheckpoint, EarlyStopping

def train(model, weights_path, encoder_input, decoder_input, decoder_output):
    checkpointer = ModelCheckpoint(filepath=weights_path, verbose=1, save_best_only=True)
    stopper = EarlyStopping(monitor='val_loss',patience=3)

    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit([encoder_input, decoder_input], decoder_output,
          batch_size=256,
          epochs=100,
          validation_split=0.2, # Keras will automatically create a validation set for us
          callbacks=[checkpointer, stopper])
BASELINE_MODEL_WEIGHTS = 'C:\\Users\\ChuuChuuPC\\hosting\\model\\baseline_model_weights.hdf5'
training_model, testing_encoder_model, testing_decoder_model = baseline_model()
if not IS_KAGGLE:
    train(training_model, BASELINE_MODEL_WEIGHTS, char_input_train, phone_input_train, phone_output_train)

def predict_baseline(input_char_seq, encoder, decoder):
    # Suppress verbose output by setting verbose=0
    state_vectors = encoder.predict(input_char_seq, verbose=0)

    prev_phone = np.zeros((1, 1, PHONE_TOKEN_COUNT))
    prev_phone[0, 0, phone_to_id[START_PHONE_SYM]] = 1.

    end_found = False
    pronunciation = ''
    while not end_found:
        # Suppress verbose output by setting verbose=0
        decoder_output, h, c = decoder.predict([prev_phone] + state_vectors, verbose=0)

        # Predict the phoneme with the highest probability
        predicted_phone_idx = np.argmax(decoder_output[0, -1, :])
        predicted_phone = id_to_phone[predicted_phone_idx]

        pronunciation += predicted_phone + ' '

        if predicted_phone == END_PHONE_SYM or len(pronunciation.split()) > MAX_PHONE_SEQ_LEN:
            end_found = True

        # Setup inputs for next time step
        prev_phone = np.zeros((1, 1, PHONE_TOKEN_COUNT))
        prev_phone[0, 0, predicted_phone_idx] = 1.
        state_vectors = [h, c]

    return pronunciation.strip()
# Helper method for converting vector representations back into words
def one_hot_matrix_to_word(char_seq):
    word = ''
    for char_vec in char_seq[0]:
        if np.count_nonzero(char_vec) == 0:
            break
        hot_bit_idx = np.argmax(char_vec)
        char = id_to_char[hot_bit_idx]
        word += char
    return word


# Some words have multiple correct pronunciations
# If a prediction matches any correct pronunciation, consider it correct.
def is_correct(word,test_pronunciation):
    correct_pronuns = phonetic_dict[word]
    for correct_pronun in correct_pronuns:
        if test_pronunciation == correct_pronun:
            return True
    return False


def sample_baseline_predictions(sample_count, word_decoder):
    sample_indices = random.sample(range(TEST_EXAMPLE_COUNT), sample_count)
    for example_idx in sample_indices:
        example_char_seq = char_input_test[example_idx:example_idx+1]
        predicted_pronun = predict_baseline(example_char_seq, testing_encoder_model, testing_decoder_model)
        example_word = word_decoder(example_char_seq)
        pred_is_correct = is_correct(example_word, predicted_pronun)
        print( '1'if pred_is_correct else '0 ', example_word,'-->', predicted_pronun)
training_model.load_weights(BASELINE_MODEL_WEIGHTS)  # also loads weights for testing models
#sample_baseline_predictions(10, one_hot_matrix_to_word)

def one_hot_encode(sequence, num_tokens):
    # Creating a zero-filled matrix of shape (sequence length, number of tokens)
    one_hot_matrix = np.zeros((len(sequence), num_tokens))

    # Setting the appropriate indices to 1 for each character in the sequence
    for i, idx in enumerate(sequence):
        one_hot_matrix[i, idx] = 1

    # Reshaping the matrix to fit the model's expected input format
    return one_hot_matrix.reshape(1, -1, num_tokens)

import numpy as np

def word_to_one_hot_matrix(word, char_to_id, max_char_seq_len, char_token_count):
    """
    Convert a word to its one-hot encoded matrix representation.

    Args:
    word (str): The word to convert.
    char_to_id (dict): Mapping from characters to their IDs.
    max_char_seq_len (int): Maximum length of character sequence.
    char_token_count (int): Total number of character tokens.

    Returns:
    numpy.ndarray: One-hot encoded matrix representation of the word.
    """
    word_matrix = np.zeros((max_char_seq_len, char_token_count))
    for t, char in enumerate(word):
        if char in char_to_id:
            char_id = char_to_id[char]
            word_matrix[t, char_id] = 1.
    return word_matrix
training_model.save("C:\\Users\\ChuuChuuPC\\hosting\\model\\baseline_model.h5")

def compare_phonetic_transcriptions(word1, word2, char_to_id, max_char_seq_len, char_token_count,phone_to_id, id_to_phone, max_phone_seq_len):
    """
    Compare the phonetic transcriptions of two words using the model.

    Args:
    word1 (str): The first word.
    word2 (str): The second word.
    ... [Other parameters: mappings and constants from your model's context]

    Returns:
    list: The phonemes the child might have trouble pronouncing.
    """


    # Convert words to one-hot matrices
    word1_matrix = word_to_one_hot_matrix(word1.upper(), char_to_id, max_char_seq_len, char_token_count)
    word2_matrix = word_to_one_hot_matrix(word2.upper(), char_to_id, max_char_seq_len, char_token_count)


    # Predict phonetic transcriptions
    predicted_pronun1 = predict_baseline(np.array([word1_matrix]), testing_encoder_model, testing_decoder_model)
    predicted_pronun2 = predict_baseline(np.array([word2_matrix]), testing_encoder_model, testing_decoder_model)

    print(predicted_pronun1)
    print(predicted_pronun2)

    # Find differing letters
    diff_letters = [phoneme1 for phoneme1, phoneme2 in zip(predicted_pronun1.split(), predicted_pronun2.split()) if phoneme1 != phoneme2]

    return diff_letters

# Example usage
word1 = "alright"  # Correct word
word2 = "aight"  # Mispronounced word
trouble_phonemes = compare_phonetic_transcriptions(
    word1, word2,
    char_to_id, MAX_CHAR_SEQ_LEN, CHAR_TOKEN_COUNT,
    phone_to_id, id_to_phone, MAX_PHONE_SEQ_LEN
)
print("The phonemes the child might have trouble pronouncing:", trouble_phonemes)


from keras.models import load_model

def load_trained_model():
    model_path = 'C:\\Users\\ChuuChuuPC\\hosting\\model\\baseline_model.h5'
    model = load_model(model_path)
    # Define or load additional parameters/components
    char_to_id = {'': 0, '.': 1, '-': 2, "'": 3, 'A': 4, 'B': 5, 'C': 6, 'D': 7, 'E': 8, 'F': 9, 'G': 10, 'H': 11, 'I': 12, 'J': 13, 'K': 14, 'L': 15, 'M': 16, 'N': 17, 'O': 18, 'P': 19, 'Q': 20, 'R': 21, 'S': 22, 'T': 23, 'U': 24, 'V': 25, 'W': 26, 'X': 27, 'Y': 28, 'Z': 29}
    max_char_seq_len = 17
    char_token_count = 30
    phone_to_id = {'': 0, '\t': 1, '\n': 2, 'AA': 3, 'AA0': 4, 'AA1': 5, 'AA2': 6, 'AE': 7, 'AE0': 8, 'AE1': 9, 'AE2': 10, 'AH': 11, 'AH0': 12, 'AH1': 13, 'AH2': 14, 'AO': 15, 'AO0': 16, 'AO1': 17, 'AO2': 18, 'AW': 19, 'AW0': 20, 'AW1': 21, 'AW2': 22, 'AY': 23, 'AY0': 24, 'AY1': 25, 'AY2': 26, 'B': 27, 'CH': 28, 'D': 29, 'DH': 30, 'EH': 31, 'EH0': 32, 'EH1': 33, 'EH2': 34, 'ER': 35, 'ER0': 36, 'ER1': 37, 'ER2': 38, 'EY': 39, 'EY0': 40, 'EY1': 41, 'EY2': 42, 'F': 43, 'G': 44, 'HH': 45, 'IH': 46, 'IH0': 47, 'IH1': 48, 'IH2': 49, 'IY': 50, 'IY0': 51, 'IY1': 52, 'IY2': 53, 'JH': 54, 'K': 55, 'L': 56, 'M': 57, 'N': 58, 'NG': 59, 'OW': 60, 'OW0': 61, 'OW1': 62, 'OW2': 63, 'OY': 64, 'OY0': 65, 'OY1': 66, 'OY2': 67, 'P': 68, 'R': 69, 'S': 70, 'SH': 71, 'T': 72, 'TH': 73, 'UH': 74, 'UH0': 75, 'UH1': 76, 'UH2': 77, 'UW': 78, 'UW0': 79, 'UW1': 80, 'UW2': 81, 'V': 82, 'W': 83, 'Y': 84, 'Z': 85, 'ZH': 86}
    id_to_phone = {0: '', 1: '\t', 2: '\n', 3: 'AA', 4: 'AA0', 5: 'AA1', 6: 'AA2', 7: 'AE', 8: 'AE0', 9: 'AE1', 10: 'AE2', 11: 'AH', 12: 'AH0', 13: 'AH1', 14: 'AH2', 15: 'AO', 16: 'AO0', 17: 'AO1', 18: 'AO2', 19: 'AW', 20: 'AW0', 21: 'AW1', 22: 'AW2', 23: 'AY', 24: 'AY0', 25: 'AY1', 26: 'AY2', 27: 'B', 28: 'CH', 29: 'D', 30: 'DH', 31: 'EH', 32: 'EH0', 33: 'EH1', 34: 'EH2', 35: 'ER', 36: 'ER0', 37: 'ER1', 38: 'ER2', 39: 'EY', 40: 'EY0', 41: 'EY1', 42: 'EY2', 43: 'F', 44: 'G', 45: 'HH', 46: 'IH', 47: 'IH0', 48: 'IH1', 49: 'IH2', 50: 'IY', 51: 'IY0', 52: 'IY1', 53: 'IY2', 54: 'JH', 55: 'K', 56: 'L', 57: 'M', 58: 'N', 59: 'NG', 60: 'OW', 61: 'OW0', 62: 'OW1', 63: 'OW2', 64: 'OY', 65: 'OY0', 66: 'OY1', 67: 'OY2', 68: 'P', 69: 'R', 70: 'S', 71: 'SH', 72: 'T', 73: 'TH', 74: 'UH', 75: 'UH0', 76: 'UH1', 77: 'UH2', 78: 'UW', 79: 'UW0', 80: 'UW1', 81: 'UW2', 82: 'V', 83: 'W', 84: 'Y', 85: 'Z', 86: 'ZH'}
    max_phone_seq_len = 18

    # Return all necessary objects
    return (model, char_to_id, max_char_seq_len, char_token_count,phone_to_id, id_to_phone, max_phone_seq_len)


