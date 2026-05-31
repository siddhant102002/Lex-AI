import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'contracts.db')

def get_connection():
    """Opens a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable dict-like access to rows
    return conn

def init_db():
    """Creates the contracts table if it doesn't exist yet."""
    conn = get_connection()
    conn.execute(
        "CREATE TABLE IF NOT EXISTS contracts ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "filename TEXT NOT NULL,"
        "contract_type TEXT NOT NULL,"
        "text TEXT NOT NULL,"
        "summary TEXT,"
        "uploaded_at TEXT NOT NULL"
        ")"
    )
    conn.commit()
    conn.close()
                 

def save_contract(filename: str, contract_type: str, text: str, summary: str = None):
    """Saves a contract to the database. Returns the new contract's id."""
    conn = get_connection()
    cursor = conn.execute("""
        INSERT INTO contracts (filename, contract_type, text, summary, uploaded_at)
        VALUES (?, ?, ?, ?, ?)
    """, (filename, contract_type, text, summary, datetime.now().isoformat()))
    conn.commit()
    contract_id = cursor.lastrowid
    conn.close()
    return contract_id


def get_all_contracts():
    """Returns all stored contracts — id, filename, type, uploaded_at only (no full text)."""
    conn = get_connection()
    contracts = conn.execute("""
        SELECT id, filename, contract_type, uploaded_at FROM contracts
        ORDER BY uploaded_at DESC
    """).fetchall()
    conn.close()
    return [dict(c) for c in contracts]


def get_contract_by_id(contract_id: int):
    """Returns a single contract including full text."""
    conn = get_connection()
    contract = conn.execute("""
        SELECT * FROM contracts WHERE id = ?
    """, (contract_id,)).fetchone()
    conn.close()
    return dict(contract) if contract else None


def get_contracts_by_type(contract_type: str):
    """Returns all contracts of a specific type — used for comparison."""
    conn = get_connection()
    contracts = conn.execute("""
        SELECT * FROM contracts WHERE contract_type = ?
        ORDER BY uploaded_at DESC
    """, (contract_type,)).fetchall()
    conn.close()
    return [dict(c) for c in contracts]
   