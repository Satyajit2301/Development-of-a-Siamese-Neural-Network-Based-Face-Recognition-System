"""
database.py
-----------
SQLite storage for registered users and their face embeddings.

Tables
  users(user_id PK, name, created_at)
  embeddings(id PK, user_id -> users.user_id ON DELETE CASCADE, embedding BLOB)

Every enrolment photo gives one 128-d float32 embedding, so a user normally
has several embeddings. The cropped face images are also saved as PNG files
(one folder per user) for display in the interface.
"""
import os
import re
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import cv2
import numpy as np

EMBEDDING_DIM = 128
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "registered_users", "users.db")
DEFAULT_IMAGES_DIR = os.path.join(BASE_DIR, "registered_users", "faces")
_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def is_valid_user_id(user_id):
    """User IDs: 1-32 characters, letters / digits / '_' / '-' only (safe as a folder name)."""
    return isinstance(user_id, str) and bool(_ID_PATTERN.match(user_id))


class UserDatabase:
    """Create, read, update and delete registered users and their embeddings."""

    def __init__(self, db_path=DEFAULT_DB_PATH, images_dir=DEFAULT_IMAGES_DIR):
        self.db_path = db_path
        self.images_dir = images_dir
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        with self._session() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS users (
                                user_id TEXT PRIMARY KEY,
                                name TEXT NOT NULL,
                                created_at TEXT NOT NULL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS embeddings (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                user_id TEXT NOT NULL
                                    REFERENCES users(user_id) ON DELETE CASCADE,
                                embedding BLOB NOT NULL)""")

    @contextmanager
    def _session(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _check_embeddings(embeddings):
        emb = np.asarray(embeddings, dtype="float32")
        if emb.ndim == 1:
            emb = emb[None, :]
        if emb.ndim != 2 or emb.shape[0] == 0 or emb.shape[1] != EMBEDDING_DIM:
            raise ValueError(f"Embeddings must have shape (n, {EMBEDDING_DIM})")
        if not np.all(np.isfinite(emb)):
            raise ValueError("Embeddings contain NaN or infinite values")
        return np.ascontiguousarray(emb)

    def _save_images(self, user_id, face_images):
        folder = os.path.join(self.images_dir, user_id)
        os.makedirs(folder, exist_ok=True)
        start = len([f for f in os.listdir(folder) if f.endswith(".png")])
        for i, img in enumerate(face_images, start=start + 1):
            cv2.imwrite(os.path.join(folder, f"face_{i:02d}.png"), img)

    # ---------------- create ----------------
    def add_user(self, user_id, name, embeddings, face_images=None):
        """Register a new user. Raises ValueError for a bad or duplicate ID."""
        if not is_valid_user_id(user_id):
            raise ValueError("User ID must be 1-32 characters: letters, digits, '_' or '-'")
        name = (name or "").strip()
        if not name:
            raise ValueError("Name must not be empty")
        emb = self._check_embeddings(embeddings)
        if self.user_exists(user_id):
            raise ValueError(f"User ID '{user_id}' is already registered")
        with self._session() as conn:
            conn.execute("INSERT INTO users (user_id, name, created_at) VALUES (?, ?, ?)",
                         (user_id, name, datetime.now().isoformat(timespec="seconds")))
            conn.executemany("INSERT INTO embeddings (user_id, embedding) VALUES (?, ?)",
                             [(user_id, e.tobytes()) for e in emb])
        if face_images is not None:
            self._save_images(user_id, face_images)

    # ---------------- read ----------------
    def user_exists(self, user_id):
        with self._session() as conn:
            row = conn.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,)).fetchone()
        return row is not None

    def get_user(self, user_id):
        """Return {user_id, name, created_at, n_embeddings} or None."""
        for u in self.list_users():
            if u["user_id"] == user_id:
                return u
        return None

    def list_users(self):
        """All registered users as a list of dictionaries (oldest first)."""
        with self._session() as conn:
            rows = conn.execute("""SELECT u.user_id, u.name, u.created_at, COUNT(e.id)
                                   FROM users u LEFT JOIN embeddings e ON e.user_id = u.user_id
                                   GROUP BY u.user_id ORDER BY u.created_at, u.user_id""").fetchall()
        return [{"user_id": r[0], "name": r[1], "created_at": r[2], "n_embeddings": r[3]} for r in rows]

    def get_all_embeddings(self):
        """Return {user_id: {"name": str, "embeddings": float32 array (n, 128)}}."""
        with self._session() as conn:
            users = conn.execute("SELECT user_id, name FROM users ORDER BY created_at, user_id").fetchall()
            rows = conn.execute("SELECT user_id, embedding FROM embeddings ORDER BY id").fetchall()
        out = {uid: {"name": name, "embeddings": []} for uid, name in users}
        for uid, blob in rows:
            out[uid]["embeddings"].append(np.frombuffer(blob, dtype="float32"))
        for uid in out:
            out[uid]["embeddings"] = np.array(out[uid]["embeddings"], dtype="float32").reshape(-1, EMBEDDING_DIM)
        return out

    def count_users(self):
        return len(self.list_users())

    # ---------------- update ----------------
    def update_name(self, user_id, new_name):
        new_name = (new_name or "").strip()
        if not new_name:
            raise ValueError("Name must not be empty")
        with self._session() as conn:
            cur = conn.execute("UPDATE users SET name = ? WHERE user_id = ?", (new_name, user_id))
        return cur.rowcount > 0

    def add_embeddings(self, user_id, embeddings, face_images=None):
        """Add more reference embeddings to an existing user."""
        if not self.user_exists(user_id):
            raise ValueError(f"User ID '{user_id}' is not registered")
        emb = self._check_embeddings(embeddings)
        with self._session() as conn:
            conn.executemany("INSERT INTO embeddings (user_id, embedding) VALUES (?, ?)",
                             [(user_id, e.tobytes()) for e in emb])
        if face_images is not None:
            self._save_images(user_id, face_images)

    # ---------------- delete ----------------
    def delete_user(self, user_id):
        """Delete a user, their embeddings and their saved face images. Returns True if deleted."""
        with self._session() as conn:
            cur = conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            deleted = cur.rowcount > 0
        folder = os.path.join(self.images_dir, user_id)
        if deleted and is_valid_user_id(user_id) and os.path.isdir(folder):
            shutil.rmtree(folder)
        return deleted
