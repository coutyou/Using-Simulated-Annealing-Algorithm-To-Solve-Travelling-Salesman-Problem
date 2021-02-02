import random
import numpy as np 
from collections import defaultdict

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from TravellingSalesmanProblemAnnealer import TravellingSalesmanProblemAnnealer


CITY_NUM = 30
ITER_TIMES = 20


def load_data(CITY_NUM):
    file_name = f'BEN{CITY_NUM}-XY.txt'
    res = {}
    with open(file_name) as f:
        lines = f.readlines()[1:-1]
        for i in range(len(lines)):
            tmp = lines[i].split()
            res[i] = (int(tmp[0]), int(tmp[1]))
    return res


def cal_distance(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5


def draw_state(state, cities, e, energy_record):
    result_x = [0 for col in range(len(state) + 1)]
    result_y = [0 for col in range(len(state) + 1)]
    
    for i in range(len(state)):
        result_x[i] = cities[state[i]][0]
        result_y[i] = cities[state[i]][1]

    result_x[-1] = result_x[0]
    result_y[-1] = result_y[0]

    plt.figure()
    plt.plot(result_x, result_y, marker='>', mec='r', mfc='w',label='Route')
    plt.legend()
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Result: {e}")

    plt.figure()
    plt.plot(range(len(energy_record)), energy_record)
    plt.xlabel("Times Of Update")
    plt.ylabel("Energy")
    plt.title(f"Energy Curve")
    
    plt.show() 


if __name__ == '__main__':
    cities = load_data(CITY_NUM)

    distance_matrix = defaultdict(dict)
    for ka, va in cities.items():
        for kb, vb in cities.items():
            distance_matrix[ka][kb] = 0.0 if kb == ka else cal_distance(va, vb)

    best_energy = 1e10
    best_state = None
    best_record = None
    es = []
    for _ in range(ITER_TIMES):
        init_state = list(cities)
        random.shuffle(init_state)

        tsp = TravellingSalesmanProblemAnnealer(init_state, distance_matrix)
        state, e, record = tsp.anneal()

        es.append(e)
        if e < best_energy:
            best_energy = e
            best_state = state
            best_record = record
        
        print()

    print(f"\nBest Result: {min(es)}    Worst Result: {max(es)}   Avg Of Results: {np.mean(es)}    Var Of Results: {np.var(es)}")
    print(f"Best State: {best_state}")
    draw_state(best_state, cities, best_energy, best_record)