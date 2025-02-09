pip install tensorflow numpy
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import deque

class PoetryGeneratorFromFile:
    def __init__(self, max_sequence_len=100, max_words=5000):
        self.max_sequence_len = max_sequence_len
        self.max_words = max_words
        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        self.model = None
    
    def load_dataset(self, filename):
        """
        Membaca dataset dari file txt
        """
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                poems = file.read().split('\n\n')  # Memisahkan puisi berdasarkan baris kosong
            
            # Membersihkan data
            poems = [poem.strip() for poem in poems if poem.strip()]
            return poems
            
        except FileNotFoundError:
            print(f"Error: File {filename} tidak ditemukan!")
            return []
        except Exception as e:
            print(f"Error membaca file: {str(e)}")
            return []
    
    def prepare_data(self, poems):
        """
        Mempersiapkan data puisi untuk training
        """
        # Tokenisasi teks
        self.tokenizer.fit_on_texts(poems)
        total_words = len(self.tokenizer.word_index) + 1
        
        # Membuat sequences
        input_sequences = []
        for poem in poems:
            token_list = self.tokenizer.texts_to_sequences([poem])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)
        
        # Padding sequences
        padded_sequences = pad_sequences(input_sequences, 
                                      maxlen=self.max_sequence_len, 
                                      padding='pre')
        
        # Membuat input dan target
        X = padded_sequences[:, :-1]
        y = padded_sequences[:, -1]
        y = tf.keras.utils.to_categorical(y, num_classes=total_words)
        
        return X, y, total_words
    
    def build_model(self, total_words, embedding_dim=100):
        """
        Membangun model LSTM
        """
        self.model = Sequential([
            Embedding(total_words, embedding_dim, 
                     input_length=self.max_sequence_len-1),
            LSTM(256, return_sequences=True),
            LSTM(256),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(total_words, activation='softmax')
        ])
        
        self.model.compile(loss='categorical_crossentropy', 
                          optimizer='adam', 
                          metrics=['accuracy'])
    
    def train(self, X, y, epochs=100, batch_size=32):
        """
        Melatih model
        """
        history = self.model.fit(X, y, 
                               epochs=epochs, 
                               batch_size=batch_size,
                               validation_split=0.1)
        return history
    
    def sample_with_temperature(self, predictions, temperature=0.7):
        """
        Menggunakan temperature sampling untuk meningkatkan kreativitas
        """
        predictions = np.asarray(predictions).astype('float64')
        predictions = np.log(predictions) / temperature
        exp_predictions = np.exp(predictions)
        predictions = exp_predictions / np.sum(exp_predictions)
        probas = np.random.multinomial(1, predictions, 1)
        return np.argmax(probas)
    
    def generate_poem(self, seed_text, next_words=50, temperature=0.7):
        """
        Menghasilkan puisi dengan mencegah pengulangan
        """
        recent_words = deque(maxlen=5)
        result = seed_text
        
        for _ in range(next_words):
            token_list = self.tokenizer.texts_to_sequences([result])[0]
            token_list = pad_sequences([token_list], 
                                     maxlen=self.max_sequence_len-1, 
                                     padding='pre')
            
            predicted = self.model.predict(token_list, verbose=0)[0]
            next_index = self.sample_with_temperature(predicted, temperature)
            
            output_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == next_index:
                    output_word = word
                    break
            
            if output_word in recent_words:
                continue
                
            recent_words.append(output_word)
            result += " " + output_word
            
            if len(recent_words) % np.random.randint(6, 9) == 0:
                result += "\n"
        
        return result

# Contoh penggunaan
if __name__ == "__main__":
    # Inisialisasi generator
    generator = PoetryGeneratorFromFile()
    
    # Baca dataset dari file
    poems = generator.load_dataset('puisi.txt')
    
    if poems:
        print(f"Berhasil memuat {len(poems)} puisi dari file")
        
        # Persiapkan data
        X, y, total_words = generator.prepare_data(poems)
        
        # Bangun dan latih model
        generator.build_model(total_words)
        history = generator.train(X, y, epochs=100)
        
        # Generate puisi baru
        seed = "Aku adalah"
        generated_poem = generator.generate_poem(seed, temperature=0.7)
        print("\nPuisi yang dihasilkan:")
        print(generated_poem)
    else:
        print("Tidak dapat melanjutkan karena error dalam membaca file")