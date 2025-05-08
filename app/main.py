# app/main.py

"""
Точка входа Streamlit-приложения.
"""

from ui import run_app

# Streamlit при выполнении скрипта присваивает __name__ == "__main__",
# поэтому run_app() вызовется сразу
if __name__ == "__main__":
    run_app()
