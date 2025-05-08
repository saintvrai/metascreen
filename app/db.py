# app/db.py

import sqlite3
import json
from datetime import datetime
import numpy as np


def init_db(db_path="ecran.db"):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    # Таблица для сохранения результатов ГА
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ga_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date TEXT,
            structure TEXT,
            params TEXT,
            se REAL,
            ghz REAL,
            refl REAL,
            absor REAL,
            pop_size REAL,
            generations REAL,
            mutation_rate REAL
        )
    ''')
    # Таблица для сохранения графиков обучения ГА
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_graphs_ga (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date TEXT,
            structure TEXT,
            graph_name TEXT,
            graph BLOB
        )
    ''')
    # Таблица для сохранения графиков обучения TF модели
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_graphs_tf (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date TEXT,
            structure TEXT,
            graph_name TEXT,
            graph BLOB
        )
    ''')
    # Таблица для сохранения обученных моделей TF
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trained_models_tf (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date TEXT,
            structure TEXT,
            model_name TEXT,
            params TEXT,
            mse REAL,
            mae REAL,
            r2 REAL,
            model_path TEXT,
            scalers_path TEXT
        )
    ''')
    conn.commit()
    return conn


def save_trained_model_to_db(conn, structure, model_name, mse, mae, r2, model_path, scalers_path,run_date, params={}):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO trained_models_tf (run_date, structure, model_name, params, mse, mae, r2, model_path, scalers_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (run_date, structure, model_name, json.dumps(params), mse, mae, r2, model_path, scalers_path))
    conn.commit()

def save_ga_result(conn, structure, params, se, ghz, refl, absor, pop_size, generations, mutation_rate, run_date):
    cursor = conn.cursor()
    params_json = json.dumps(params.tolist() if isinstance(params, (list, np.ndarray)) else params)
    cursor.execute('''
        INSERT INTO ga_results (run_date, structure, params, se, ghz, refl, absor, pop_size, generations, mutation_rate)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (run_date, structure, params_json, float(se), float(ghz), float(refl), float(absor), pop_size, generations, mutation_rate))
    conn.commit()

def save_graph_to_db(conn, table, structure, graph_bytes, graph_name, run_date):
    cursor = conn.cursor()
    cursor.execute(f'''
        INSERT INTO {table} (run_date, structure, graph_name, graph)
        VALUES (?, ?, ?, ?)
    ''', (run_date, structure, graph_name, graph_bytes))
    conn.commit()

def get_training_graphs(conn, table, structure=None):
    """
    Если structure указан — вернёт только графики для этой структуры,
    иначе — все.
    Возвращает список кортежей (run_date, graph_name, graph_blob).
    """
    cursor = conn.cursor()
    if structure is None:
        # возвращаем ВСЕ графики сразу с полем structure
        sql = f"""
            SELECT run_date, structure, graph_name, graph
            FROM {table}
            ORDER BY run_date DESC
        """
        cursor.execute(sql)
        return cursor.fetchall()

    else:
        # возвращаем только для конкретной структуры, без колонки structure
        sql = f"""
            SELECT run_date, graph_name, graph
            FROM {table}
            WHERE structure = ?
            ORDER BY run_date DESC
        """
        cursor.execute(sql, (structure,))
        return cursor.fetchall()