import numpy as np
import librosa
from tensorflow.keras.models import Model
import pickle
from sklearn.metrics.pairwise import cosine_similarity


# Funkcja dzieli audio na któtkie nagrania o podanej długości
def split_audio_to_slices(audio, seconds):
    sr = 16000  # Ustawienie częstotliwości próbkowania na 16 kHz
    signal, sr = librosa.load(audio, sr=sr)  # Wczytanie pliku audio i przeskalowanie do zadanej częstotliwości

    list_for_parts = []  # Lista do przechowywania fragmentów audio
    len_of_signal = int(len(signal))  # Obliczenie długości całego sygnału w próbkach
    step = seconds * sr  # Przeliczenie długości fragmentu na liczbę próbek
    
    # Pętla dzieli sygnał audio na fragmenty o zadanej długości
    for i in np.arange(start=0, stop=len_of_signal-step, step=step):
        # Wycinanie fragmentu od próbki `i` do `i + step` i dodanie do listy
        list_for_parts.append(signal[i:i+step].tolist())

    list_for_parts = np.array(list_for_parts)  # Konwersja listy fragmentów na tablicę NumPy

    return list_for_parts  # Zwrócenie tablicy fragmentów
    


# Klasa służąca do ekstrakcji embeddingów
class EmbeddingExtractor:
    
    def __init__(self, model, bottleneck, scaler_before_embedding, scaler_after_embedding=None, lda=None):
        self.model = model  # Przypisanie modelu do obiektu
        self.bottleneck = bottleneck # Nazwa warstwy bottleneck
        self.scaler_before_embedding = scaler_before_embedding  # Skaler do standaryzacji MFCC przed generowaniem embeddingów
        self.scaler_after_embedding = scaler_after_embedding  # Skaler do standaryzacji embeddingów przed zastosowaniem LDA
        self.lda = lda  # Przypisanie modelu LDA do obiektu

    # Funkcja obliczająca współczynniki MFCC dla danego nagrania audio
    def calucate_MFCC(self, audio):
        quantity_of_mel_coef = 40  # Liczba współczynników MFCC
        quantity_of_mel_filters = 60  # Liczba filtrów Mel

        # Obliczanie współczynników MFCC za pomocą librosa
        mfcc = librosa.feature.mfcc(y=audio, 
                                    sr=16000, 
                                    n_mfcc=quantity_of_mel_coef, 
                                    n_mels=quantity_of_mel_filters).T
        
        #Oczliczam pochodne aby dane były zgodne z tymi na których model był trenowany
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        mfcc = np.hstack([mfcc, delta, delta2])

        # Standaryzacja MFCC przed generowaniem embeddingów
        mfcc = self.scaler_before_embedding.transform(mfcc)
        return mfcc

    # Funkcja obliczająca embedding na podstawie wcześniej przetworzonych MFCC
    def calcuate_embedding(self, audio_MFCC):
        
        intermediate_layer_model = Model(inputs=self.model.inputs,
                                         outputs=self.model.get_layer(self.bottleneck).output)
        intermediate_output = intermediate_layer_model.predict(audio_MFCC[np.newaxis, ...])
        
        return intermediate_output

    # Funkcja postprocessingu embeddingu – standaryzacja i LDA, jeżeli nie przekażemy skalera i LDA to nie zostanie przeprowadzony postprocessig
    def transform_audio_postprocessing(self, embedding):
        if self.scaler_after_embedding is not None:
            embedding = self.scaler_after_embedding.transform(embedding)  # Standaryzacja embeddingu
        if self.lda is not None:
            embedding = self.lda.transform(embedding)  # Użycie LDA
        
        return embedding

    # Funkcja łączy wszystkie poprzednie 
    def process_audio_to_embedding(self, audio):
        mfcc = self.calucate_MFCC(audio)
        embedding = self.calcuate_embedding(mfcc)
        embedding = self.transform_audio_postprocessing(embedding)

        return embedding
    



 # Funckja przeprowadza pełny proces tworzenia embeddingu
def create_embedding(audio, EmbExtr, seconds):

    # Podział audio na fragmenty
    splited_audio = split_audio_to_slices(audio, seconds)

    # Generowanie embeddingów z krókich fragmentów audio
    sum_of_embeddings = np.zeros(128)
    for one_part in splited_audio:
        oneS_embedding = EmbExtr.process_audio_to_embedding(one_part)[0]
        sum_of_embeddings += oneS_embedding

    embedding_enrollment = sum_of_embeddings / len(splited_audio)

    return embedding_enrollment
    

# Funkcja służy do weryfikacji użytkownika 
def verification(embedding_enrollment, embedding_test):
    embedding_enrollment = embedding_enrollment.reshape(1, -1)
    embedding_test = embedding_test.reshape(1, -1)
    
    # Wybieram próg odrzucenia i liczę podobieństwo cos pomiędzy nagraniami
    threshold = 0.6
    cos_sim = cosine_similarity(embedding_enrollment, embedding_test)[0][0]

    # Sprawdzam czy score jest powyżej progu odrzucenia
    is_that_person = cos_sim > threshold

    return is_that_person, cos_sim
