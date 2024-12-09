import sqlite3

# Połączenie z bazą danych
def get_connection():
    conn = sqlite3.connect('embedding.db', check_same_thread=False)
    return conn

# Funkcja do inicjalizacji bazy danych
def initialize_database():
    conn = get_connection()
    cursor = conn.cursor()


    # Tworzenie tabeli użytkowników
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        surname TEXT NOT NULL
    )
    ''')


    cursor.execute('''
    CREATE TABLE IF NOT EXISTS embeddings (
        id_embedding INTEGER PRIMARY KEY AUTOINCREMENT,
        id_user INTEGER NOT NULL,
        emb TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(id_user) REFERENCES users(id)
    )
    ''')
    # Zapisanie zmian
    conn.commit()
    conn.close()