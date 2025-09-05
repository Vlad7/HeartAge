# Age groups
import math
import pandas as pd
import openpyxl
import enum
#from IPython.display import display
import numpy as np
import wfdb
import HiguchiFractalDimension.hfd
import csv
import matplotlib.pyplot as plt
import os.path
import sys
import os
# Импортируем функцию bwr
import bwr

import bwr
#import nbimporter
import pan_tompkins as pt

import neurokit2 as nk
import numpy as np
import wfdb
import HiguchiFractalDimension.hfd
import csv
import matplotlib.pyplot as plt
import scipy.stats as stats
from biosppy import storage
from biosppy.signals import ecg

#Folder with files in each there is RR-intervals time series
rr_intervals_folder="rr_intervals/all"



######################################################################################################
################################### LOAD RR-INTERVALS TIME SERIES ####################################
######################################################################################################
def list_files_with_rr_intervals():
    """Get list of files with rr_intervals time series from rr_interval/all folder"""
    import os

    directory = rr_intervals_folder

    # Фильтрация только файлов
    files = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]

    for file in files:
        print(file)

    return files


def extract_from_files_rr_time_series(files):
    """Extract from files RR intervals time series

        input: files - file names
        output: rr_time_series_dictionary - dictionary with id as key and list of RR-intervals as value"""

    import re

    rr_time_series_dictionary = {}

    for file in files:
        filename = file

        # Используем регулярное выражение для извлечения числового индекса
        match = re.search(r'_(\d+)\.txt', filename)
        if match:
            index = match.group(1)
            rr_time_series_dictionary[index] = None
            #print("Индекс:", index)

            file_path = rr_intervals_folder + "/" + file
            # Чтение файла, начиная со второй строки
            with open(file_path, "r") as file:
                # Пропускаем первую строку
                next(file)

                # Читаем остальные строки
                rr_intervals = [line.strip() for line in file]

            # Вывод значений
            #for rr in rr_intervals:
            #    print(rr)

            rr_intervals = [int(float(x)) for x in rr_intervals]
            #print(rr_intervals)
            rr_time_series_dictionary[index] = rr_intervals

        else:
            print("Индекс не найден")

    #print(rr_time_series_dictionary)

    return rr_time_series_dictionary


def write_spectral_calculated_values_to_csv(hfd_of_ecg_1, age_indexes_for_id, age_ranges_for_id):
    # ECG 1 and 2 simulationusly

    with open('output/spectral_calculated.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for key in age_indexes_for_id.keys():
            spamwriter.writerow([key, age_indexes_for_id[key], age_ranges_for_id[key], localize_floats(hfd_of_ecg_1[key])])

        """, RECORD.DATABASE[type_of_ecg_cut][key].Sex,
                                 RECORD.DATABASE[type_of_ecg_cut][key].BMI]"""


"""
import numpy as np
from scipy import signal, interpolate

def hrv_psd_welch(rr_ms, t_rr_s, fs=4.0):
    rr = np.asarray(rr_ms, dtype=float)
    t  = np.asarray(t_rr_s, dtype=float)

    # 1) очищенные NN и их время уже на входе
    # 2) равномерная сетка
    t_uniform = np.arange(t[0], t[-1], 1.0/fs)
    # интерполяция тахограммы (RR как функция времени)
    f = interpolate.CubicSpline(t, rr)
    rr_uniform = f(t_uniform)

    # 3) детренд
    rr_uniform = signal.detrend(rr_uniform, type='linear')

    # 4) PSD по Welch
    nperseg = int(120 * fs)  # 120-сек сегменты
    noverlap = nperseg // 2
    fHz, Pxx = signal.welch(rr_uniform, fs=fs, window='hamming',
                            nperseg=nperseg, noverlap=noverlap, detrend=False)

    return fHz, Pxx  # PSD в (мс^2/Гц) если rr_ms в мс

def bandpower(fHz, Pxx, fmin, fmax):
    idx = (fHz >= fmin) & (fHz <= fmax)
    # интеграл трапецией
    return np.trapz(Pxx[idx], fHz[idx])

# --- пример использования ---
# rr_ms: массив NN-интервалов в миллисекундах
# t_rr_s: нарастающее время появления каждого интервала в секундах (например, сумма rr_ms/1000)
# rr_ms, t_rr_s = ...

fHz, Pxx = hrv_psd_welch(rr_ms, t_rr_s, fs=4.0)

LF = bandpower(fHz, Pxx, 0.04, 0.15)
HF = bandpower(fHz, Pxx, 0.15, 0.40)
TOT = bandpower(fHz, Pxx, 0.04, 0.40)

LF_norm = 100 * LF / (LF + HF) if (LF + HF) > 0 else np.nan
HF_norm = 100 * HF / (LF + HF) if (LF + HF) > 0 else np.nan
LF_HF   = LF / HF if HF > 0 else np.nan

print(f"LF (ms^2): {LF:.1f}")
print(f"HF (ms^2): {HF:.1f}")
print(f"Total 0.04–0.40 (ms^2): {TOT:.1f}")
print(f"LFnu: {LF_norm:.1f}%, HFnu: {HF_norm:.1f}%")
print(f"LF/HF: {LF_HF:.2f}")

"""

import neurokit2 as nk
import numpy as np

def spectral_analysis(rr_time_series):
    # Пример: искусственные RR-интервалы в мс
    #rr_ms = np.array([800, 810, 790, 805, 815, 820, 780, 795, 805, 810])
    rr_ms = np.array(rr_time_series)
    # В реальной задаче это будут ваши NN-интервалы (без артефактов)

    # Переводим в "сигнал": список R-пиков во времени (в секундах)
    # Допустим, запись началась с 0:
    t_rr = np.cumsum(rr_ms) / 1000  # секунды
    rpeaks = {"ECG_R_Peaks": t_rr}

    # HRV-анализ со спектральными метриками
    results = nk.hrv(rpeaks, sampling_rate=1000, show=True)

    #print(results[["HRV_LF", "HRV_HF", "HRV_LFHF", "HRV_VLF"]])

    LF = results["HRV_LF"]
    HF = results["HRV_HF"]
    VLF = results["HRV_VLF"]
    LF_divide_HF = results["HRV_LFHF"]

    HRV_LFn = results["HRV_LFn"]
    HRV_HFn = results["HRV_HFn"]




    print(f"LF (ms^2): {LF:.1f}")
    print(f"HF (ms^2): {HF:.1f}")
    print(f"VLF (ms^2): {VLF:.1f}")
    print(f"Total 0.04–0.40 (ms^2): {LF+HF:.1f}")
    print(f"LFnu: {HRV_LFn:.1f}%, HFnu: {HRV_HFn:.1f}%")
    print(f"LF/HF: {LF_divide_HF:.2f}")

# Get list of files with rr_intervals time series
files = list_files_with_rr_intervals()

# Extract RR intervals time series from files to dictionary with id as key and RR intervals time series as value
rr_time_series_dictionary = extract_from_files_rr_time_series(files)

for key in rr_time_series_dictionary.keys():
    rr_time_series = rr_time_series_dictionary[key]



