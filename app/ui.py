# app/ui.py

import os
import json
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime
from constants import STRUCTURES, DISPLAY_NAMES, CONSTRAINTS, PARAM_ORDER, GA_DEFAULTS
from utils import patch_selectbox, parse_params_string
from ml import load_model_and_scalers, predict_em_properties
from db import (
    init_db,
    save_ga_result,
    save_trained_model_to_db,
    save_graph_to_db,
    get_training_graphs,
)

from ga import (
    random_individual,
    load_initial_population,
    mutate,
    crossover,
    fitness_batch,
    reintroduce_diversity
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import joblib
import random

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def run_app():
    # Патчим selectbox для отображения человекочитаемых названий
    patch_selectbox()

    # Инициализируем базу и создаём папки для моделей
    conn = init_db()
    for s in STRUCTURES:
        os.makedirs(f"./models/{s}", exist_ok=True)

    st.title("🛡️ Прогнозирование ЭМ характеристик метаэлемента")

    # Для сохранённых результатов ГА и загруженных вручную параметров
    if 'ga_result' not in st.session_state:
        st.session_state.ga_result = None
    if 'loaded_params' not in st.session_state:
        st.session_state.loaded_params = {}

    tabs = st.tabs([
        "📈 Нейросеть",
        "🎯 Генетический алгоритм",
        "🧠 Обучение модели",
        "🖼️ Графики ГА",
        "🖼️ Графики TF"
    ])

    # ===== TAB 1: Нейросеть =====
    with tabs[0]:
        st.subheader("📈 Расчёт по H, K, L, P нейросетью")

        structure = st.selectbox(
            "Форма метаэлемента", STRUCTURES, key="tab1_structure"
        )
        bounds = CONSTRAINTS[structure]
        model, x_scaler, y_scaler = load_model_and_scalers(structure)

        # Загрузка параметров из строки
        load_text = st.text_input(
            "📥 Вставьте '[H K L P]'",
            placeholder="[0.5 1.0 3.0 0.2]"
        )
        # Кнопка «Загрузить»
        if st.button("🔄 Загрузить параметры", key="tab1_load"):
            # 1) Проверяем, что строка не пустая
            if not load_text.strip():
                st.error("❗ Вы не вставили ни одного параметра")
            else:
                # 2) Пытаемся распарсить
                try:
                    params = parse_params_string(load_text)
                    st.session_state.loaded_params = params
                    st.success("✅ Параметры загружены")
                except Exception as e:
                    st.error(f"Ошибка формата: {e}")

        # Значения по умолчанию
        defaults = st.session_state.loaded_params
        H = st.number_input("H", *bounds['H'], value=defaults.get('H', sum(bounds['H'])/2), format="%.6f")
        K = st.number_input("K", *bounds['K'], value=defaults.get('K', sum(bounds['K'])/2), format="%.6f")
        L = st.number_input("L", *bounds['L'], value=defaults.get('L', sum(bounds['L'])/2), format="%.6f")
        P = st.number_input("P", *bounds['P'], value=defaults.get('P', sum(bounds['P'])/2), format="%.6f")

        if st.button("📊 Рассчитать", key="tab1_predict"):
            try:
                se, ghz, refl, absor = predict_em_properties(model, x_scaler, y_scaler, H, K, L, P)
                st.write(f"**ЭЭ**: {se:.4f} дБ")
                st.write(f"**Частота**: {ghz:.4f} ГГц")
                st.write(f"**Отражение**: {refl:.4f}")
                st.write(f"**Поглощение**: {absor:.4f}")
            except Exception as e:
                st.error(f"Ошибка предсказания: {e}")

        # Показываем результат из ГА, если есть
        if st.session_state.ga_result:
            st.markdown("---")
            st.markdown("#### 💾 Результат ГА")
            r = st.session_state.ga_result
            st.write(f"H,K,L,P: {r['params']}")
            st.write(f"ЭЭ: {r['se']:.4f} дБ, Freq: {r['ghz']:.4f} ГГц")

    # ===== TAB 2: Генетический алгоритм =====
    with tabs[1]:
        st.subheader("🎯 Подбор параметров ГА")

        structure = st.selectbox("Форма", STRUCTURES, key="tab2_structure")
        bounds = CONSTRAINTS[structure]
        model, x_scaler, y_scaler = load_model_and_scalers(structure)

        # Таблица прошлых запусков
        df = pd.read_sql_query(
            "SELECT * FROM ga_results WHERE structure=? ORDER BY run_date ASC",
            conn, params=(structure,)
        )
        if not df.empty:
            df['params'] = df['params'].apply(json.loads)
            df[['H','K','L','P']] = pd.DataFrame(df['params'].tolist(), index=df.index)
            st.dataframe(df.drop(columns=['id','params']))

        # Цели
        target_freq = st.number_input("Целевая частота (ГГц)", 0.0, 100.0, 10.0, 0.01)
        target_se   = st.number_input("Целевая ЭЭ (дБ)",      0.0, 100.0, 30.0, 0.01)

        # Параметры ГА
        with st.expander("⚙️ Настройки ГА"):
            pop_size      = st.slider("Популяция",    20, 300, GA_DEFAULTS['pop_size'],    10)
            generations   = st.slider("Поколения",    50, 500, GA_DEFAULTS['generations'],   10)
            mutation_rate = st.slider("Мутация",     0.01, 0.5, GA_DEFAULTS['mutation_rate'], 0.01)

        if st.button("🧬 Запустить ГА", key="tab2_run"):
            try:
                st.write(f"Поп={pop_size}, Покол={generations}, μ={mutation_rate:.2f}")
                param_order = PARAM_ORDER

                # Подготовка начальной популяции
                file = f"./{structure}/cst_data.txt"
                init_cands = load_initial_population(file, target_se, target_freq, bounds, param_order)
                population = init_cands[:pop_size//3]
                while len(population) < pop_size:
                    population.append(random_individual(bounds))

                # История и прогресс
                best_se_hist, best_freq_hist, fit_hist = [], [], []
                best_inds = []
                found_gen = None
                bar = st.progress(0)

                for gen in range(generations):
                    bar.progress((gen+1)/generations)

                    fits, se_vals, freq_vals = fitness_batch(
                        population, model, x_scaler, y_scaler,
                        target_se, target_freq,
                        w_se=1.0, w_freq=1.5,
                        generation=gen, max_generations=generations
                    )

                    # запомним историю лучшей особи по фитнесу
                    idx_best = int(np.argmax(fits))
                    best_inds.append(population[idx_best].copy())
                    best_se_hist.append(se_vals[idx_best])
                    best_freq_hist.append(freq_vals[idx_best])
                    fit_hist.append(fits[idx_best])

                    # ранний выход: оба условия
                    if abs(freq_vals[idx_best] - target_freq) <= 0.01 and se_vals[idx_best] >= target_se:
                        found_gen = gen + 1
                        st.info(
                            f"⏹️ Целевые требования достигнуты на {found_gen}-м поколении: "
                            f"SE={se_vals[idx_best]:.2f} дБ, Freq={freq_vals[idx_best]:.4f} ГГц"
                        )
                        break

                    # элитизм + скрещивания
                    elites = [population[i] for i in np.argsort(fits)[-int(0.1*pop_size):]]
                    new_pop = elites.copy()
                    while len(new_pop) < pop_size:
                        p1, p2 = random.choice(population), random.choice(population)
                        c1, c2 = crossover(p1, p2, bounds)
                        new_pop += [
                            mutate(c1, bounds, mutation_rate, gen, generations),
                            mutate(c2, bounds, mutation_rate, gen, generations)
                        ]
                    population = new_pop[:pop_size]

                    # диверсификация
                    population, _ = reintroduce_diversity(population, fits, gen, mutation_rate, bounds)

                # --- после окончания цикла поколений ---

                # 1) Выбор поколения с наибольшей SE при допуске по частоте
                tol = 0.01
                candidates = [
                    (i, se, freq)
                    for i, (se, freq) in enumerate(zip(best_se_hist, best_freq_hist))
                    if abs(freq - target_freq) <= tol
                ]
                if candidates:
                    best_gen, best_se, best_freq = max(candidates, key=lambda x: x[1])
                else:
                    # если ни одно не прошло по допуску — берём просто максимальную SE
                    best_gen = int(np.argmax(best_se_hist))
                    best_se, best_freq = best_se_hist[best_gen], best_freq_hist[best_gen]

                # финальный лучший индивид
                best_ind = best_inds[best_gen]
                se, ghz, refl, absor = predict_em_properties(
                    model, x_scaler, y_scaler, *best_ind
                )

                # Вывод результата
                st.success("🎯 Оптимальные параметры найдены:")
                st.write(f"• H,K,L,P = {np.round(best_ind, 6).tolist()}")
                st.write(f"• ЭЭ = {se:.2f} дБ")
                st.write(f"• Freq = {ghz:.4f} ГГц")
                st.write(f"• Reflection = {refl:.4f}, Absorption = {absor:.4f}")

                # Сохраняем в сессии и в БД
                st.session_state.ga_result = {
                    'params': list(best_ind), 'se': se, 'ghz': ghz,
                    'refl': refl, 'absor': absor
                }
                save_ga_result(conn, structure, best_ind, se, ghz, refl, absor,
                            pop_size, generations, mutation_rate)

                # 🖼 Построение графиков эволюции

                # 1) Эволюция SE и частоты
                fig, ax1 = plt.subplots(figsize=(10, 5))
                ax1.set_xlabel("Поколение")
                ax1.set_ylabel("ЭЭ, дБ")
                ax1.plot(
                    range(1, len(best_se_hist) + 1),
                    best_se_hist,
                    label="ЭЭ лучшей особи",
                    marker='o', markersize=3
                )
                ax1.grid(True)

                ax2 = ax1.twinx()
                ax2.set_ylabel("Частота, ГГц")
                ax2.plot(
                    range(1, len(best_freq_hist) + 1),
                    best_freq_hist,
                    label="Частота лучшей особи",
                    marker='s', markersize=3, linestyle='--'
                )
                ax2.ticklabel_format(style='plain', useOffset=False)
                ax2.axhline(
                    y=target_freq, color='green', linestyle=':',
                    label=f'Целевая частота: {target_freq:.2f} ГГц'
                )

                # рисуем «звёздочки» в лучшем поколении
                x_star     = best_gen + 1
                y_se_star   = best_se_hist[best_gen]
                y_freq_star = best_freq_hist[best_gen]

                ax1.scatter(
                    [x_star], [y_se_star],
                    color='red', s=100, marker='*',
                    label=f'Итоговая ЭЭ: {y_se_star:.2f} дБ'
                )
                ax2.scatter(
                    [x_star], [y_freq_star],
                    color='purple', s=100, marker='*',
                    label=f'Итоговая Freq: {y_freq_star:.4f} ГГц'
                )

                fig.suptitle("Эволюция ГА: ЭЭ и частота")
                fig.tight_layout()

                # легенда из обоих осей
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(
                    lines1 + lines2,
                    labels1 + labels2,
                    loc='upper left', bbox_to_anchor=(0.1, 0.95)
                )
                st.pyplot(fig)

                # 2) Эволюция фитнес-функции
                fig2, ax3 = plt.subplots(figsize=(10, 5))
                ax3.set_xlabel("Поколение")
                ax3.set_ylabel("Значение фитнес-функции")
                ax3.plot(
                    range(1, len(fit_hist) + 1),
                    fit_hist,
                    label="Фитнес лучшей особи",
                    marker='o', markersize=3
                )
                ax3.grid(True)
                fig2.suptitle("Эволюция значения фитнес-функции")
                fig2.tight_layout()
                ax3.legend()
                st.pyplot(fig2)

                # Сохраняем граф в БД
                buf = BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                try:
                    save_graph_to_db(conn, "training_graphs_ga", structure,
                    buf.read(), "GA Evolution")                    
                    st.success(f"График сохранён в БД")
                except Exception as db_e:
                    st.error(f"Не удалось сохранить график в БД: {db_e}")

            except Exception as e:
                st.error(f"Ошибка ГА: {e}")


    # ===== TAB 3: Обучение модели =====
    with tabs[2]:
        st.subheader("🧠 Обучение новой модели")
        structure = st.selectbox("Форма", STRUCTURES, key="tab3_structure")
        input_file = f"./{structure}/cst_data.txt"

        batch_size = st.number_input("Размер батча при обучении", 16, 256, 64, 16)
        epochs     = st.number_input("Максимальное количество эпох при обучении",     10, 500, 200, 10)
        log_out    = st.empty()

        if st.button("🚀 Запустить обучение", key="tab3_train"):
            try:
                # Загрузка данных
                with open(input_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                X = np.array([[i['H'], i['K'], i['L'], i['P']] for i in data], np.float32)
                Y = np.array([[i['SE'], i['GHz'], i['Reflection'], i['Absorption']] for i in data], np.float32)

                # Разбиение и масштабирование
                X_tr, X_val, Y_tr, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
                xs, ys = StandardScaler(), StandardScaler()
                X_tr_s, X_val_s = xs.fit_transform(X_tr), xs.transform(X_val)
                Y_tr_s, Y_val_s = ys.fit_transform(Y_tr), ys.transform(Y_val)

                # Уникальный идентификатор для модели и скейлеров
                unique_id    = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path   = f"./models/{structure}/model_{unique_id}.keras"
                scalers_path = f"./models/{structure}/scalers_{unique_id}.pkl"

                # Построение модели
                model = tf.keras.Sequential([
                    tf.keras.layers.Dense(256, input_shape=(4,)),
                    tf.keras.layers.LeakyReLU(0.1),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dense(128),
                    tf.keras.layers.Dense(64),
                    tf.keras.layers.Dense(32),
                    tf.keras.layers.Dense(4),
                ])
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(),
                    loss="mean_squared_error",
                    metrics=['accuracy', 'mean_absolute_error', 'mean_squared_error']
                )

                # Callback для логов и ранней остановки по точности
                class StreamlitLogger(tf.keras.callbacks.Callback):
                    def __init__(self, output_element):
                        super().__init__()
                        self.output_element = output_element
                        self.log_text = ""
                    def on_epoch_end(self, epoch, logs=None):
                        acc     = logs.get('accuracy', 0) * 100
                        val_acc = logs.get('val_accuracy', 0) * 100
                        loss    = logs.get('loss', 0)
                        self.log_text += (
                            f"Эпоха {epoch+1}: loss={loss:.4f}, "
                            f"Точность на обучении={acc:.2f}%, Точность на валидации={val_acc:.2f}%\n"
                        )
                        self.output_element.text_area(
                            "📜 Лог обучения", value=self.log_text, height=300
                        )

                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_accuracy',
                        mode='max',
                        patience=10,
                        restore_best_weights=True,
                        verbose=1
                    ),
                    tf.keras.callbacks.ModelCheckpoint(
                        filepath=model_path,
                        monitor='val_accuracy',
                        mode='max',
                        save_best_only=True,
                        verbose=1
                    ),
                    StreamlitLogger(log_out)
                ]

                # Обучение
                history = model.fit(
                    X_tr_s, Y_tr_s,
                    validation_data=(X_val_s, Y_val_s),
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=callbacks,
                    verbose=0
                )

                # Оценка и сохранение модели и скейлеров
                pred_s = model.predict(X_val_s)
                pred   = ys.inverse_transform(pred_s)
                mse    = mean_squared_error(Y_val, pred)
                mae    = mean_absolute_error(Y_val, pred)
                r2     = r2_score(Y_val, pred)
                model.save(model_path)
                joblib.dump((xs, ys), scalers_path)
                
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                save_trained_model_to_db(conn, structure, unique_id, mse, mae, r2, model_path, scalers_path, run_date=now)

                st.success("✅ Обучение завершено")
                st.write(f"Итоговая точность: {history.history['accuracy'][-1]*100:.2f}%, "
                         f"Валидационная точность: {history.history['val_accuracy'][-1]*100:.2f}%")
                st.write(f"MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

                # Построение объединённого графика метрик
                fig, axes = plt.subplots(3, 1, figsize=(10, 18))
                # — Accuracy
                axes[0].plot([x * 100 for x in history.history['accuracy']],     label='Train Accuracy')
                axes[0].plot([x * 100 for x in history.history['val_accuracy']], label='Val Accuracy')
                axes[0].set_title('График точности при обучении , %')
                axes[0].set_xlabel('Эпоха')
                axes[0].set_ylabel('Точность, %')
                axes[0].legend()
                # — MSE (loss)
                axes[1].plot(history.history['loss'],     label='Train MSE')
                axes[1].plot(history.history['val_loss'], label='Val MSE')
                axes[1].set_title('Среднеквадратичная ошибка (MSE)')
                axes[1].set_xlabel('Эпоха')
                axes[1].set_ylabel('MSE')
                axes[1].legend()
                # — MAE
                axes[2].plot(history.history['mean_absolute_error'],     label='Train MAE')
                axes[2].plot(history.history['val_mean_absolute_error'], label='Val MAE')
                axes[2].set_title('Средняя абсолютная ошибка (MAE)')
                axes[2].set_xlabel('Эпоха')
                axes[2].set_ylabel('MAE')
                axes[2].legend()

                fig.tight_layout()
                st.pyplot(fig)

                # Сохранение графика в БД
                buf = BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                
                try:
                    save_graph_to_db(conn, "training_graphs_tf", structure, buf.read(), f"train_metrics_{unique_id}", run_date= now)
                    st.success(f"График сохранён в БД")
                except Exception as db_e:
                    st.error(f"Не удалось сохранить график в БД: {db_e}")

            except Exception as e:
                st.error(f"Ошибка обучения: {e}")
                
        # ===== TAB 4: Сохранённые графики ГА =====
    with tabs[3]:
        st.subheader("🖼️ Сохранённые графики оптимизации ГА")
        # выбор структуры
        structure_filter = st.selectbox(
            "Выберите форму метаэлемента",
            STRUCTURES,
            key="tab4_structure"
        )

        # 1) достаём из БД все графики ГА (с полем structure внутри)
        all_graphs = get_training_graphs(conn, "training_graphs_ga")
        # 2) фильтруем по выбранной структуре
        graphs = [
            (run_date, name, blob)
            for run_date, struct, name, blob in all_graphs
            if struct == structure_filter
        ]

        # 3) если графиков нет — предупреждаем
        if not graphs:
            st.info(f"Для структуры **{DISPLAY_NAMES[structure_filter]}** ещё нет сохранённых графиков ГА.")
        else:
            # 4) группируем их по дате
            graph_dict = {}
            for run_date, name, blob in graphs:
                graph_dict.setdefault(run_date, []).append((name, blob))

            # 5) вытягиваем из ga_results все строчки по этой структуре
            df = pd.read_sql_query(
                "SELECT * FROM ga_results WHERE structure=? ORDER BY run_date DESC",
                conn, params=(structure_filter,)
            )

            if df.empty:
                st.info("Хотя графики-то есть, но результатов GA ещё нет. Запустите оптимизацию.")
            else:
                # распаковываем JSON-параметры
                df["params"] = df["params"].apply(json.loads)
                df[["H","K","L","P"]] = pd.DataFrame(df["params"].tolist(), index=df.index)

                # для каждой строки делаем блок с метриками и чекбоксом
                for idx, row in df.iterrows():
                    st.markdown("---")
                    st.markdown(f"**🕒 Дата:** `{row['run_date']}`")
                    st.markdown(
                        f"**SE:** {row['se']:.4f} дБ | "
                        f"**Freq:** {row['ghz']:.4f} ГГц | "
                        f"**Refl:** {row['refl']:.4f} | "
                        f"**Absor:** {row['absor']:.4f}"
                    )
                    st.markdown(
                        f"**Параметры H,K,L,P:** "
                        f"{row['H']:.4f}, {row['K']:.4f}, "
                        f"{row['L']:.4f}, {row['P']:.4f}"
                    )

                    show = st.checkbox("📈 Показать графики этого запуска", key=f"show_ga_{idx}")
                    if show:
                        for name, blob in graph_dict.get(row["run_date"], []):
                            st.image(blob, caption=f"{name} ({row['run_date']})", use_container_width=True)


    # ===== TAB 5: Сохранённые графики TF =====
    with tabs[4]:
        st.subheader("🖼️ Сохранённые графики модели TF")
        # выбор структуры
        structure_filter_tf = st.selectbox(
            "Выберите форму метаэлемента",
            STRUCTURES,
            key="tab5_structure"
        )

        # 1) вытаскиваем все графики TF
        all_graphs_tf = get_training_graphs(conn, "training_graphs_tf")
        # 2) фильтруем по структуре
        graphs_tf = [
            (run_date, name, blob)
            for run_date, struct, name, blob in all_graphs_tf
            if struct == structure_filter_tf
        ]

        if not graphs_tf:
            st.info(f"Для структуры **{DISPLAY_NAMES[structure_filter_tf]}** ещё нет сохранённых графиков TF.")
        else:
            # группируем по дате
            graph_dict_tf = {}
            for run_date, name, blob in graphs_tf:
                graph_dict_tf.setdefault(run_date, []).append((name, blob))

            # 3) вытягиваем метрики из trained_models_tf
            tf_df = pd.read_sql_query(
                "SELECT * FROM trained_models_tf WHERE structure=? ORDER BY run_date DESC",
                conn, params=(structure_filter_tf,)
            )

            if tf_df.empty:
                st.info("Метрик моделей TF ещё нет. Обучите хотя бы одну модель.")
            else:
                for idx, row in tf_df.iterrows():
                    st.markdown("---")
                    st.markdown(f"**🕒 Дата:** {row['run_date']}")
                    st.markdown(
                        f"**MSE:** {row['mse']:.4f} | "
                        f"**MAE:** {row['mae']:.4f} | "
                        f"**R²:** {row['r2']:.4f}"
                    )

                    show = st.checkbox("📈 Показать графики этого запуска", key=f"show_tf_{idx}")
                    if show:
                        for name, blob in graph_dict_tf.get(row["run_date"], []):
                            st.image(blob, caption=f"{name} ({row['run_date']})", use_container_width=True)


if __name__ == "__main__":
    run_app()
