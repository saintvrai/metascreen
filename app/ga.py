# app/ga.py

import json
import pandas as pd
import numpy as np
import random
import numpy as np
from constants import PARAM_ORDER

def random_individual(bounds: dict, param_order=PARAM_ORDER) -> np.ndarray:
    """
    Создаёт одного случайного индивида в рамках bounds.
    """
    return np.array([
        np.random.uniform(bounds[p][0], bounds[p][1])
        for p in param_order
    ])



def load_initial_population(
    input_file: str,
    target_se: float,
    target_freq: float,
    bounds: dict,
    param_order=PARAM_ORDER,
    freq_tolerance: float = 0.15
) -> list[np.ndarray]:
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df = df.rename(columns={'Reflection':'Refl', 'Transmission':'Absor'})

        # вместо dff = df[mask] сделаем явное .loc[...] + .copy()
        mask = (
            (df['GHz'] >= target_freq - freq_tolerance) &
            (df['GHz'] <= target_freq + freq_tolerance)
        )
        dff = df.loc[mask].copy()     # ← .copy() чтобы избежать SettingWithCopyWarning
        if dff.empty:
            dff = df.sample(min(20, len(df)))

        # теперь можно безопасно создавать новые колонки
        dff['f_diff']  = (dff['GHz'] - target_freq).abs()
        dff['se_diff'] = (dff['SE']  - target_se).abs()

        dff = dff.sort_values(['f_diff','se_diff'])
        candidates = []
        for _, row in dff.iterrows():
            vals = []
            ok = True
            for p in param_order:
                v = row.get(p)
                if v is None or not (bounds[p][0] <= v <= bounds[p][1]):
                    ok = False
                    break
                vals.append(v)
            if ok:
                candidates.append(np.array(vals, dtype=float))
        return candidates

    except Exception as e:
        print(f"[GA] load_initial_population error: {e}")
        return []


def mutate(
    ind: np.ndarray,
    bounds: dict,
    rate: float,
    generation: int,
    max_generations: int,
    param_order=PARAM_ORDER
) -> np.ndarray:
    """
    Адаптивная мутация: в начале сильнее, к концу слабее.
    """
    new = ind.copy()
    strength = max(0.2, 1.0 - generation/max_generations)
    for i, p in enumerate(param_order):
        if np.random.rand() < rate:
            span = bounds[p][1] - bounds[p][0]
            if np.random.rand() < 0.7:
                sigma = span * 0.2 * strength
                delta = np.random.normal(0, sigma)
            else:
                delta = (2*np.random.rand()-1) * span * strength
            new[i] = np.clip(ind[i] + delta, bounds[p][0], bounds[p][1])
    return new


def crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    bounds: dict,
    param_order=PARAM_ORDER
) -> tuple[np.ndarray, np.ndarray]:
    """
    Несколько стратегий кроссовера, возвращает двух потомков.
    """
    mode = np.random.choice(['one_point','two_point','uniform','blend'])
    size = len(p1)

    if mode == 'one_point':
        pt = np.random.randint(1, size)
        c1 = np.concatenate([p1[:pt], p2[pt:]])
        c2 = np.concatenate([p2[:pt], p1[pt:]])

    elif mode == 'two_point':
        pts = sorted(np.random.choice(range(1, size), 2, replace=False))
        c1 = np.concatenate([p1[:pts[0]], p2[pts[0]:pts[1]], p1[pts[1]:]])
        c2 = np.concatenate([p2[:pts[0]], p1[pts[0]:pts[1]], p2[pts[1]:]])

    elif mode == 'uniform':
        c1, c2 = p1.copy(), p2.copy()
        mask = np.random.rand(size) < 0.5
        c1[mask], c2[mask] = p2[mask], p1[mask]

    else:  # blend (BLX-α)
        alpha = 0.3
        c1 = np.zeros(size)
        c2 = np.zeros(size)
        for i, p in enumerate(param_order):
            mn, mx = min(p1[i], p2[i]), max(p1[i], p2[i])
            span = mx - mn
            low = max(bounds[p][0], mn - alpha*span)
            high = min(bounds[p][1], mx + alpha*span)
            c1[i] = np.random.uniform(low, high)
            c2[i] = np.random.uniform(low, high)

    return c1, c2


def fitness_batch(
    population: list[np.ndarray],
    model,
    x_scaler,
    y_scaler,
    target_se: float,
    target_freq: float,
    w_se: float = 1.0,
    w_freq: float = 1.5,
    generation: int = 0,
    max_generations: int = 200
) -> tuple[list[float], np.ndarray, np.ndarray]:
    """
    Вычисляет fitness для всех индивидов, возвращает
    (fitness_list, se_values, freq_values).
    """
    X = np.array(population, dtype=float)
    Xs = x_scaler.transform(X)
    preds_s = model.predict(Xs)
    preds = y_scaler.inverse_transform(preds_s)
    fits, se_vals, f_vals = [], preds[:,0], preds[:,1]

    # динамические веса
    prog = generation/max_generations
    phase = min(1.0, 2*prog)
    wf = w_freq * (1 - 0.7*phase)
    ws = w_se   * (0.5 + 0.5*phase)

    for se, freq in zip(se_vals, f_vals):
        # бонус за точную частоту
        fm = np.exp(-50*(freq-target_freq)**2)
        sf = 1/(1+np.exp(-0.5*(se-target_se)))
        fit = fm*1000*wf + sf*100*ws + min(20, max(0,se-target_se))*5*ws
        fits.append(fit)
    return fits, se_vals, f_vals


def reintroduce_diversity(
    population: list[np.ndarray],
    fitness_vals: list[float],
    generation: int,
    mutation_rate: float,
    bounds: dict,
    param_order=PARAM_ORDER,
    diversity_threshold: float = 0.05
) -> tuple[list[np.ndarray], bool]:
    """
    Если популяция слишком однородна — вводим новых особей.
    Возвращает (new_population, diversity_added_flag).
    """
    arr = np.stack(population)
    stds = np.std(arr, axis=0)
    norm = np.mean(stds / (arr.max(axis=0)-arr.min(axis=0)+1e-10))
    if norm < diversity_threshold:
        # 10% элиты
        idx = np.argsort(fitness_vals)[-max(1, len(population)//10):]
        elite = [population[i] for i in idx]
        new = []
        for _ in range(len(population)-len(elite)):
            if random.random()<0.5:
                par = random.choice(elite)
                new.append(mutate(par, bounds, rate=mutation_rate, generation=generation, max_generations=generation+1))
            else:
                new.append(random_individual(bounds, param_order))
        return elite + new, True
    return population, False
