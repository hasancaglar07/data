# db.py - Ayrı dosya olarak kaydedin
import sqlite3
import os

DB_FILE = "qa_database.db"

def init_db():
    """Veritabanını başlat, yoksa oluştur."""
    if not os.path.exists(DB_FILE):
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS qa_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

def save_qa(question, answer):
    """Soru ve cevabı DB'ye kaydet."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO qa_sessions (question, answer) VALUES (?, ?)
    """, (question, answer))
    conn.commit()
    conn.close()

def get_all_qa():
    """Tüm soru-cevapları getir."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT question, answer, timestamp FROM qa_sessions ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows