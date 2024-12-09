from fastapi import FastAPI, File, UploadFile, Form, Query, Path
import json
from pydantic import BaseModel
from typing import List
import io
from tools_to_creating_embeddings import create_embedding, verification, EmbeddingExtractor
from DB_service import get_connection, initialize_database
import pytz
from datetime import datetime
import numpy as np
from tensorflow.keras.models import load_model
import pickle


# Inicjalizacja bazy danych (tworzenie tabel, jeśli nie istnieją)
initialize_database()

# Tworzenie instancji aplikacji FastAPI
app = FastAPI()

# Uzyskiwanie globalnego połączenia do bazy danych
conn = get_connection()
cursor = conn.cursor()


# Definicje modeli danych
class User(BaseModel):
    """Model użytkownika"""
    name: str
    surname: str

class Embedding(BaseModel):
    """Model embeddingu przypisanego do użytkownika"""
    id_user: int
    id_embedding: int 
    emb: List[float]




# Wczytuję potrzbne rzeczy do stworzenia ekstraktora embeddingu
model_path = "C:/Users/zbugo/Desktop/praktyki_zadania/20/modele_itp/model_GRU.h5"
model = load_model(model_path)
    
scaler_path = "C:/Users/zbugo/Desktop/praktyki_zadania/20/modele_itp/scaler.pkl"
with open(scaler_path, "rb") as file:
    scaler_before_embedding = pickle.load(file)

# Tworzę instancję klasy, będzie to nasz ekstraktor embeddingów 
EmbExtr = EmbeddingExtractor(
    model=model,
    bottleneck="bottleneck",  # Nazwa warstwy "bottleneck" w modelu
    scaler_before_embedding=scaler_before_embedding
)






### Endpointy API ###






@app.post("/create_user/{id}")
def create_user(id: int, user: User):
    """
    Endpoint do dodawania nowego użytkownika.
    """
    
    # Połączenie z bazą danych
    with get_connection() as conn:
        cursor = conn.cursor()
    
    # Pobranie istniejących ID użytkowników
    cursor.execute('SELECT id FROM users')
    user_ids = [row[0] for row in cursor.fetchall()]

    # Sprawdzenie, czy użytkownik już istnieje
    if id in user_ids:
        return {"message": "Użytkownik o danym ID już istnieje"}


    # Dodanie użytkownika do tabeli "users"
    cursor.execute('''
        INSERT INTO users (id, name, surname)
        VALUES (?, ?, ?)
    ''', (id, user.name, user.surname))
    conn.commit()

    return {"message": "Użytkownik został dodany pomyślnie"}








@app.put("/add_embedding/{id}")
async def add_embedding(id: int = Path(...), file: UploadFile = File(...)):
    """
    Endpoint do dodawania embeddingu dla użytkownika.
    """
    
    # Połączenie z bazą danych
    with get_connection() as conn:
        cursor = conn.cursor()

    
    
    # Pobranie kolumy id z tabeli users
    cursor.execute('SELECT id FROM users')
    user_ids = [row[0] for row in cursor.fetchall()]

    # Pobranie kolumny id_user z tabeli embeddings
    cursor.execute('SELECT id_user FROM embeddings')
    user_ids_from_emb_table = [row[0] for row in cursor.fetchall()]

    # Sprawdzenie czy użytkownik w ogóle ma profil
    if id not in user_ids:
        return {"message": "Osoba o danym ID nie ma jeszcze profilu"}
    
    
    # Sprawdzenie czy isntieje embedding danego użytkownika, jeżeli tak, to go usuwam aby za chwilę stworzyć nowy embedding enrollment
    embedding_exist = False
    if id in user_ids_from_emb_table:
        cursor.execute("DELETE FROM embeddings WHERE id_user = ?", (id,))
        conn.commit()
        embedding_exist = True

    # Wczytanie pliku audio
    audio = await file.read()
    audio = io.BytesIO(audio)

    # Tworzę embedding enrollment
    embedding_enrollment = create_embedding(audio, EmbExtr, 1)

    # Ustawienie polskiej strefy czasowej
    polish_timezone = pytz.timezone("Europe/Warsaw")

    # Pobierz bieżący czas w polskiej strefie czasowej
    current_time = datetime.now(polish_timezone).strftime("%Y-%m-%d %H:%M:%S")

    # Dodanie embeddingu do tabeli "embeddings"
    cursor.execute('''
        INSERT INTO embeddings (id_user, emb, created_at)
        VALUES (?, ?, ?)
    ''', (id, json.dumps(embedding_enrollment.tolist()), current_time))
    conn.commit()

    if embedding_exist: 
        return {"message": f"Embedding został zakutalizowany dla użytkownika: {id}"}
    else :
        return {"message": f"Pomyślnie dodano embedding dla użytkownika: {id}"}

    






@app.put("/verification/{your_id}")
async def verification_of_user(your_id: int = Path(..., description="Podaj swoje ID"), test_file: UploadFile = File(...)):
    """
    Endpoint do weryfiakcji uzytkowników.
    """

    # Połączenie z bazą danych 
    with get_connection() as conn:
        cursor = conn.cursor()
    
    
    # Pobranie kolumny id z tabeli users
    cursor.execute('SELECT id FROM users')
    user_ids = [row[0] for row in cursor.fetchall()]

    # Pobranie kolumny id_users z tabeli embeddings
    cursor.execute('SELECT id_user FROM embeddings')
    user_ids_from_emb_table = [row[0] for row in cursor.fetchall()]


    # Sprawdzenie czy dana osoba w ogóle ma profil
    if your_id not in user_ids:
        return {"message": "Osoba o danym ID nie ma jeszcze profilu"}
    

    # Sprawdzenie czy dana osoba ma stworzony embedding
    if your_id not in user_ids_from_emb_table:
        return {"message": "Osoba o danym ID nie ma jeszcze dodanego embeddingu"}


    # Pobranie embeddingu danej osoby    
    cursor.execute("SELECT emb FROM embeddings WHERE id_user = ?", (your_id,))
    embedding_enrollment = [row[0] for row in cursor.fetchall()][0]
    
    # Parsowanie JSON na tablicę numpy
    embedding_enrollment = np.array(json.loads(embedding_enrollment))
    embedding_enrollment = embedding_enrollment.reshape(1, -1)


    # Wczytanie pliku audio
    audio = await test_file.read()
    audio = io.BytesIO(audio)

    # Tworzenie embeddingu testowego
    embedding_test = create_embedding(audio, EmbExtr, 1)

    # Obliczanie podobieństwa cos pomiędzy nagraniami enrollment a test aby przeprowadzic weryfikację
    is_that_person, score = verification(embedding_enrollment, embedding_test)


    return {"message": f":{is_that_person}, score: {score}"}
    







@app.get("/get_user/{id}")
def get_user(id: int = Path(...)):
    """
    Endpoint do sprawdzenia ostaniej daty modyfikacji embeddingu.
    """
    # Połączenie się z bazą danych
    with get_connection() as conn:
        cursor = conn.cursor()

    # Pobranie kolumny id_user z tabeli embeddings
    cursor.execute('SELECT id_user FROM embeddings')
    user_ids = [row[0] for row in cursor.fetchall()]


    # Sprawdzenie czy użytkownik ma stworzony embedding
    if id not in user_ids:
        return {"message": "Użytkownik albo nie ma jeszcze założonego konta, albo nie stworzył swojego embeddingu"}
    
    # Pobranie daty edytowania bądź tworzenia embeddingu
    cursor.execute("SELECT created_at FROM embeddings WHERE id_user = ?", (id,))
    date_of_changes = cursor.fetchone()

    return {"message": f"Ostatnia zmiana (utworzenie bądź akutalizacja) embeddingu miała miejsce: {date_of_changes[0]}"}








@app.delete("/delete_user/{id}")
def delete_user(id: int = Path(...)):
    """
    Endpoint do usuwania użytkownika i jego embeddingu.
    """
    # Połączenie z bazą danych
    with get_connection() as conn:
        cursor = conn.cursor()

    # Pobranie istniejących ID użytkowników i embeddingów
    cursor.execute('SELECT id FROM users')
    user_ids = [row[0] for row in cursor.fetchall()]

    # Pobranie kolumny id_user z tabeli embeddings
    cursor.execute('SELECT id_user FROM embeddings')
    user_ids_from_embedding_table = [row[0] for row in cursor.fetchall()]

    # Sprawdzenie czy użytkownik ma przypisany embedding
    if id in user_ids_from_embedding_table:
        # Usuwanie zarówno użytkownika, jak i embeddingu
        cursor.execute("DELETE FROM embeddings WHERE id_user = ?", (id,))
        cursor.execute("DELETE FROM users WHERE id = ?", (id,))
        conn.commit()
        return {"message": f"Pomyślnie usunięto użytkownika {id} oraz jego embedding"}
    
    #Sprawdzenie czy użytkownik ma konto
    elif id in user_ids:
        # Usuwanie samego użytkownika
        cursor.execute("DELETE FROM users WHERE id = ?", (id,))
        conn.commit()
        return {"message": f"Pomyślnie usunięto użytkownika o ID {id}, brak powiązanego embeddingu"}
    else:
        return {"message": "Użytkownik o danym ID nie istnieje"}
