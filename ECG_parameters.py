"""import wfdb
from wfdb import processing

path_to_dataset_folder = 'D:/SCIENCE/Datasets/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0'

sig, fields = wfdb.rdsamp(r'{0}/0001'.format(path_to_dataset_folder), channels=[0])
xqrs = processing.XQRS(sig=sig[:,0], fs=fields['fs'])
xqrs.detect()
wfdb.plot_items(signal=sig, ann_samp=[xqrs.qrs_inds])
"""
import wfdb
import csv
import os
import numpy as np
from biosppy.signals import ecg
import matplotlib.pyplot as plt
from statistics import mean

from neurokit2 import rsp_amplitude
from wfdb import processing
import neurokit2 as nk
import pandas as pd

path_to_dataset_folder = 'D:/SCIENCE/Datasets/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0'
#path_to_dataset_folder  = 'C:/Datasets/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0'

csv_info_file = 'subject-info.csv'

rr_intervals_folder="rr_intervals/all"

DATABASE_ATTRIBUTES = []

age_groups = {'NaN': 'none',
              '1': '18 - 19',
                  '2': '20 - 24',
                  '3': '25 - 29',
                  '4': '30 - 34',
                  '5': '35 - 39',
                  '6': '40 - 44',
                  '7': '45 - 49',
                  '8': '50 - 54',
                  '9': '55 - 59',
                  '10': '60 - 64',
                  '11': '65 - 69',
                  '12': '70 - 74',
                  '13': '75 - 79',
                  '14': '80 - 84',
                  '15': '85 - 92',
                  }

def breaked_ECGs():
    """ECG's with breakes (empties in ECG line)"""

    # Id's of breacked first ecg's
    breaked_first_ecg_ids = []
    # Id's of breacked second ecg's
    breaked_second_ecg_ids = []

    with open('breaked_list/breaked_first_ecg.txt', 'r', encoding='utf-8') as file:
        for str in file:
            breaked_first_ecg_ids.append(str.strip())  # strip - remove spaces and '\n'

    with open('breaked_list/breaked_second_ecg.txt', 'r', encoding='utf-8') as file:
        for str in file:
            breaked_second_ecg_ids.append(str.strip())

    print("Breaked_first_ecg_ids: ", breaked_first_ecg_ids)
    print("Breaked_second_ecg_ids: ", breaked_second_ecg_ids)

    return breaked_first_ecg_ids, breaked_second_ecg_ids


def sets_with_breaked_ECGs(breaked_first_ecg_ids, breaked_second_ecg_ids):
    """General for two ECG's and unique of each ECG"""

    # General for two lists
    general = list(set(breaked_first_ecg_ids) & set(breaked_second_ecg_ids))  # Пересечение множеств
    general.sort()
    print("General: ", general)

    first_unique = list(set(breaked_first_ecg_ids) - set(general))  # First unique from general
    second_unique = list(set(breaked_second_ecg_ids) - set(general))  # Second unique from general
    first_unique.sort()
    second_unique.sort()
    print("First unique: ", first_unique)
    print("Second unique: ", second_unique)

    return general, first_unique, second_unique

def open_record_wfdb(id, min_point, max_point,   remotely):
    """Open record with wfdb"""
    record = None

    if remotely:
        record = wfdb.rdrecord(id, min_point, max_point, [0, 1], pn_dir='autonomic-aging-cardiovascular')
    else:
        record = wfdb.rdrecord(
            path_to_dataset_folder + '/' + id, min_point, max_point, [0, 1])

    return record

#######################################################################################################################
#################################### EXTRACTING RR INTERVALS ##########################################################
#######################################################################################################################
def extract_cleaned_signal_and_R_peaks(signal, sampling_rate, show_graphics):
    """BIO SPPY library for extracting cleaned signal and R-peaks from ECG signal
        input:
            signal - ECG signal
            sampling_rate - sampling_rate
            show_graphics - show graphics

        output:

            filtered_signal, r_peaks - filtered signal and R peaks time series

    """

    # signal, mdata = storage.load_txt('./examples/ecg.txt')

    # Додання 500 відліків зліва та справа для коректного подальшого розпізнання R-піків
    extended_signal = np.pad(signal, (500, 500), mode='edge')

    # Для фільтрації ЕКГ, виявлення R-піків та побудови графіків використовується бібліотека biosppy
    out = ecg.ecg(signal=extended_signal, sampling_rate=sampling_rate, show=show_graphics)

    # Відфільтрований сигнал
    filtered_extended = out['filtered']

    # Відступаємо від початку 500 і від кінця 500 (обернена операція до np.pad)
    # Чомусь на графіку в точці 500 не співпадає з початком обрізаного.
    filtered_original = filtered_extended[500:-500]


    ################################ PRINT CLEANED SIGNAL ############################################

    #Maybe error!
    print("CLEANED SIGNAL:")
    print(filtered_original)

    ###################################################################################################

    # Отримання індексів R-піків
    r_peaks = out['rpeaks']

    # Відступаємо назад на 500 для індексів R-піків (обернена операція до np.pad)
    r_peaks = r_peaks - 500

    # Дополнительно: сохранение в файл
    #np.savetxt("rr_peaks/peaks_{0}.txt".format(id), r_peaks,
    #           header="Peaks (ms)", comments='', fmt="%.6f")

    return filtered_original, r_peaks

def calculate_RR_intervals(id, r_peaks):
    """Calculate and save RR-intervals time series for each id
        input:
            id - id of record
            r_peaks - time series of r_peaks
    """
    # Вычисляем R-R интервалы (в милисекундах)
    # rr_intervals = np.diff(r_peaks)

    # np.savetxt("rr_intervals/rr_intervals_{0}.txt".format(id), rr_intervals,
    #           header="RR Intervals (ms)", comments='', fmt="%.6f")

def open_record(id, min_point, max_point, remotely):

    """ Open each record with ECGs by Id

        Input parameters:
            - Id - id of record
            - min_point - minimum point, at which starts ECG (including this point)
            - max_point - maximum point, at which ends ECG (not including this point)

        Output parameters:
            - [sequence_1, sequence_2] - list with sequence_1 for first ECG and sequence_2 for second ECG

            Describing:
                wfdb.rdrecord(path + '/' + id, min_point, max_point, [0, 1])

                min_point = 0 - The starting sample number to read for all channels
                                (point from what graphic starts (min_point)).

                max_point = None - The sample number at which to stop reading for all
                channels (max_point). Reads the entire duration by default.

                [0, 1] - first two channels (ECG 1, ECG 2); [0] - only first ECG.
            """

    record = None

    if min_point < 0:
        print("Too low minimal point of ECG! Now minimal point is 0!")
        min_point = 0

    if os.path.isfile(path_to_dataset_folder + '/' + id + '.hea') or os.path.isfile(path_to_dataset_folder + '/' + id + '.dat'):
        try:
            record = open_record_wfdb(id, min_point, max_point, remotely)

        except:
            max_point = None
            record = open_record_wfdb(id, min_point, max_point, remotely)
            print("Too hight maximal point of ECG! Now maximal point is None!")
    else:
        print("File with record doesn't exist!")
        return None

    #display(record.__dict__)


    sequence_1 = []
    sequence_2 = []


    # print(record.p_signal)

    for x in record.p_signal:

        # Use first ECG
        sequence_1.append(x[0])

        # Use second ECG
        sequence_2.append(x[1])


    print("Length of first ECG with id {0}: {1}".format(id, str(len(sequence_1))))
    print("Length of second ECG with id {0}: {1}".format(id, str(len(sequence_2))))


    return [sequence_1, sequence_2]

def read_ECGs_annotation_data(is_remotely, except_breaked):
    """ Open csv info file, print header and information for each record.

        input: is_remotely - download annotation file and record remotely from internet

    """
    files = list_files_with_rr_intervals()
    # Ecg's with indexes that have variability
    ids_with_variability = extract_from_files_ids(files)
    print(ids_with_variability)
    # Path to CSV file with annotation
    path = ""

    # Id's of breaked first and second ecg's
    breaked_first_ecg_ids, breaked_second_ecg_ids = breaked_ECGs()

    # Id's general both for first ecg, first unique, second unique
    general, first_unique, second_unique = sets_with_breaked_ECGs(breaked_first_ecg_ids, breaked_second_ecg_ids)

    ECGs_features_male = {}
    ECGs_features_female = {}

    # Check, if dataset is remotely located
    if is_remotely:
        path = csv_info_file
    else:
        path = path_to_dataset_folder + '/' + csv_info_file

    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        # Получаем первую строку для инициализации DATABASE_ATTRIBUTES
        first_row = next(csv_reader)

        # Setting counter for first row
        line_count = 0

        for col in first_row:
            DATABASE_ATTRIBUTES.append(col)

        # Union of general and first_unique (breaked first ECG)
        breaked_first_ecg = list(set(general) | set(first_unique))
        breaked_first_ecg.sort()

        # Обрабатываем оставшиеся строки
        for row in csv_reader:

            line_count += 1

            # Check, if ECG in first ecg breaked list
            if (row[0] in breaked_first_ecg):
                continue

            if (row[0] not in ids_with_variability):
                continue
            # 780 - 800; 1081 < !!!! 42



            if (line_count < 852):
                continue

            print ("Hello")
            # If Id is not available
            if (row[0] == 'NaN'):
                continue

            # If age category is not available. For future maybe do self-organizing without ages.
            if (row[1] == 'NaN'):
                continue

            else:
                # Open record returns ecg_1 and ecg_2
                ecg_s = open_record(row[0], 0, 480501, remotely=is_remotely)

                if ecg_s is None:
                    continue

                # Signal with first ECG
                ecg_signal = np.array(ecg_s[0])

                # Частота дискретизації
                sampling_rate = 1000

                #To show graphics
                show_graphics = True

                # Filter signal to cleaned and detect r_peaks
                cleaned_signal, r_peaks = extract_cleaned_signal_and_R_peaks(ecg_signal,
                                               sampling_rate, show_graphics)


                ########################## Візуалізація відфільтрованого сигналу з R-піками ###########################

                if (show_graphics):
                    print(cleaned_signal)
                    print("Signal length:", len(cleaned_signal))

                    print("First and last R-peaks:", r_peaks[0], r_peaks[-1])

                    import matplotlib.pyplot as plt
                    print(f"R-peaks: {r_peaks}")
                    print(f"Cleaned signal length: {len(cleaned_signal)}")
                    plt.plot(cleaned_signal)

                    valid_r_peaks = r_peaks[r_peaks < len(cleaned_signal)]

                    plt.scatter(valid_r_peaks, cleaned_signal[valid_r_peaks], color='red')
                    plt.show()

                #######################################################################################################

                import biosppy

                # Обробка ЕКГ
                #out = biosppy.signals.ecg.ecg(signal=ecg_signal, sampling_rate=500, show=True)
                #t = biosppy.signals.ecg.getTPositions(ecg_proc=out, show=True)
                # R-пики: out['rpeaks']

                #for x in t[2]:
                #    print(x)

                # Margin 200 for correct delineation
                margin = 200  # or adjust based on your delineation window size
                valid_r_peaks = r_peaks[(r_peaks > margin) & (r_peaks < len(cleaned_signal) - margin)]


                # Next, use NeuroKit for P, Q, S, T (around R)
                # Delineate the ECG signal using neurokit2, cwt with hight precision, for quicker use dwt
                _, waves_peaks = nk.ecg_delineate(cleaned_signal, valid_r_peaks,
                                                 sampling_rate=sampling_rate, method="cwt", show=show_graphics)


                isoline, waves, features = calculate_ECG_features(cleaned_signal, r_peaks, waves_peaks)

                #waves, features = calculate_ECG_features(cleaned_signal, r_peaks, waves_peaks)



                id = row[0]
                age_category = row[1]

                if row[2] == '0':
                    #ECG dictionary with id as key and list as value with age category, sex, ECG features
                    write_ECG_parameters_to_csv('male', id, age_category, features)

                if row[2] == '1':
                    # ECG dictionary with id as key and list as value with age category, sex, ECG features
                    write_ECG_parameters_to_csv('female', id, age_category, features)

                # Припустимо, ми аналізуємо перші тридцять серцевих циклів на графіку:

                count_plot = 100
                if show_graphics:
                    cleaned_signal = cleaned_signal - isoline
                    plot_ECG_features(cleaned_signal, waves, count_plot)










                """
                import matplotlib.pyplot as plt
                import numpy as np


                plt.figure(figsize=(12, 4))
                plt.plot(time, ecg_signal, label="ECG")
                plt.scatter(time[r_peaks], ecg_signal[r_peaks], color='red', label="R-peaks")
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude")
                plt.title("ECG with R-peaks")
                plt.legend()
                plt.grid(True)
                plt.show()


                # === Отобразим
                nk.ecg_plot(signals)
                plt.show()"""
                """
                import neurokit2 as nk
                import matplotlib.pyplot as plt

                # 1. Генеруємо синтетичний ЕКГ-сигнал
                #signal = nk.ecg_simulate(duration=10, sampling_rate=1000)

                # 2. Аналіз сигналу: виявлення компонентів (P, QRS, T)
                ecg_signals, info = nk.ecg_process(signal, sampling_rate=1000)

                # 3. Отримання індексів початку і кінця компонентів
                r_peaks = info["ECG_R_Peaks"]
                p_peaks = info["ECG_P_Peaks"]
                t_peaks = info["ECG_T_Peaks"]

                # 4. Візуалізація
                plt.figure(figsize=(15, 5))
                plt.plot(ecg_signals["ECG_Clean"], label="ECG Signal")

                # Позначимо точки на графіку
                plt.scatter(p_peaks, ecg_signals["ECG_Clean"][p_peaks], color="green", label="P peaks")
                plt.scatter(r_peaks, ecg_signals["ECG_Clean"][r_peaks], color="red", label="R peaks")
                plt.scatter(t_peaks, ecg_signals["ECG_Clean"][t_peaks], color="purple", label="T peaks")

                plt.title("ЕКГ-сигнал з позначеними P, R, T")
                plt.xlabel("Час (мс)")
                plt.ylabel("Амплітуда")
                plt.legend()
                plt.grid()
                plt.show()
                """
                # Частота дискретизації
                #sampling_rate = 1000

                #r_peaks, rr_intervals = extract_cleaned_signal_and_R_peaks(signal, sampling_rate, row[0])

                """
                import csv

                ecg_attributes = [
                    {"time": 0.0, "amplitude": 0.1, "heart_rate": 75},
                    {"time": 0.01, "amplitude": 0.12, "heart_rate": 75},
                    {"time": 0.02, "amplitude": 0.14, "heart_rate": 76},
                    # и так далее...
                ]

                with open("ecg_data.csv", mode="w", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=["time", "amplitude", "heart_rate"])
                    writer.writeheader()
                    writer.writerows(ecg_attributes)
                """


import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import wfdb


def bandpass_filter(signal_data, fs, lowcut=0.5, highcut=40, order=3):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return signal.filtfilt(b, a, signal_data)


def find_r_peaks(ecg_signal, fs):
    # Можно заменить на Pan-Tompkins или WFDB аннотацию
    distance = int(0.6 * fs)  # минимум 600 мс между пиками
    peaks, _ = signal.find_peaks(ecg_signal, distance=distance, height=np.mean(ecg_signal) + 0.3 * np.std(ecg_signal))
    return peaks


def estimate_isoline2(segment):
    return np.average(segment)

"""
def find_p_wave_end(segment, isoline, threshold=0.02):
    # Находит, где сигнал "возвращается" к изолинии
    diff = np.abs(segment - isoline)
    for i in range(len(diff) - 1, -1, -1):
        if diff[i] > threshold:
            return i
    return None
"""
def find_p_wave_end(segment, isoline, threshold=0.02):
    # Находит, где сигнал "возвращается" к изолинии

    diff = segment - isoline
    for i in range(len(diff) - 1 - 50, 0, -1):
        if diff[i] > threshold:
            return i
    return None

def analyze_p_wave(signal_data, fs, r_peaks, window_before_r=0.2):
    p_ends = []
    for r in r_peaks:
        start = int(r - window_before_r * fs)
        if start < 0:
            continue
        segment = signal_data[start:r]

        # Итерации
        isoline = estimate_isoline2(segment)
        for _ in range(3):
            p_end = find_p_wave_end(segment, isoline)
            if p_end is not None:
                sub_segment = segment[p_end:]
                isoline = estimate_isoline2(sub_segment)

        p_ends.append(start + p_end if p_end else None)

    return p_ends







def detect_s_peaks(r_peaks, ecg_signal):
    """Detection S peaks"""
    s_peaks = []
    s_peaks_points = []
    for r_peak in r_peaks:

        relative = 0
        index = r_peak
        point = ecg_signal[index]

        # Go to the point under or equal isoline
        while point >= 0:
            relative += 1
            index = r_peak + relative
            point = ecg_signal[index]

        while True:
            index = r_peak + relative
            if ecg_signal[index + 1] - ecg_signal[index] < 0:
                relative+=1
            else:
                point = ecg_signal[index]
                break
        s_peaks.append(index)
        s_peaks_points.append(point)

    return s_peaks

def estimate_isoline(ecg_signal, p_end_indices, q_start_indices):
    """
    Оценивает уровень изолинии по сегментам PQ.

    ecg_signal: массив амплитуд ЭКГ
    p_end_indices: индексы конца зубца P
    q_start_indices: индексы начала зубца Q
    """
    p_end_indices = p_end_indices.astype(int)
    q_start_indices = q_start_indices.astype(int)

    isoline_values = []

    for p_end, q_start in zip(p_end_indices, q_start_indices):
        if q_start > p_end:
            segment = ecg_signal[p_end:q_start]
            if len(segment) > 0:
                mean_value = np.mean(segment)
                isoline_values.append(mean_value)

    if isoline_values:
        global_isoline = np.mean(isoline_values)
        return global_isoline
    else:
        return 0.0  # если не найдено подходящих сегментов

def calculate_HCF(r_peaks):
    # Вычисляем временные интервалы между пиками R

    intervals = np.diff(r_peaks) # Находим разности между последовательными пиками

    # Рассчитываем средний интервал (в милисекундах)
    average_interval = np.mean(intervals)

    # Рассчитываем частоту сердечных сокращений (в ударах в минуту)
    if average_interval > 0:
        heart_rate = 60 / average_interval
    else:
        heart_rate = 0

    print(f"Частота сердечных сокращений: {heart_rate:.2f} уд/мин")

    return heart_rate

def mean_RR_interval(r_peaks):
    intervals = np.diff(r_peaks)  # Находим разности между последовательными пиками

    # Рассчитываем средний интервал (в секундах)
    average_interval = np.mean(intervals)

    return average_interval

def mean_ST_interval(s_waves, t_waves):
    st = t_waves - s_waves
    average_st = np.mean(st)

    return average_st

def mean_QRS_complex(q_waves, s_waves):
    qrs = s_waves - q_waves
    average_qrs = np.mean(qrs)

    return average_qrs

def calculate_ECG_features(cleaned_signal, r_peaks, waves_peaks):

    p_start_waves = pd.Series(waves_peaks["ECG_P_Onsets"])
    p_peaks = pd.Series(waves_peaks["ECG_P_Peaks"])
    p_end_waves   = pd.Series(waves_peaks["ECG_P_Offsets"])
    q_waves = pd.Series(waves_peaks["ECG_Q_Peaks"])
    r_peaks = pd.Series(r_peaks)
    s_waves = pd.Series(waves_peaks["ECG_S_Peaks"])
    t_start_waves = pd.Series(waves_peaks["ECG_T_Onsets"])
    t_peaks = pd.Series(waves_peaks["ECG_T_Peaks"])
    t_end_waves =   pd.Series(waves_peaks["ECG_T_Offsets"])

    cycles = []

    # Задать временное окно вокруг R-пика, в котором ищем другие пики (в секундах или отсчётах)
    window = 300  # например, +/-150 отсчётов

    for r in r_peaks.dropna():
        r = int(r)

        # Найти ближайшие P, Q, S, T пики в пределах окна
        p_start = p_start_waves[(p_start_waves >= r - window) & (p_start_waves < r)].dropna()
        p_peak = p_peaks[(p_peaks >= r - window) & (p_peaks < r)].dropna()
        p_end = p_end_waves[(p_end_waves >= r - window) & (p_end_waves < r)].dropna()

        q = q_waves[(q_waves >= r - 200) & (q_waves < r)].dropna()  # Q ближе
        s = s_waves[(s_waves > r) & (s_waves <= r + 200)].dropna()

        t_start = t_start_waves[(t_start_waves > r) & (t_start_waves <= r + window + 300)].dropna()
        t_peak = t_peaks[(t_peaks > r) & (t_peaks <= r + window + 300)].dropna()
        t_end = t_end_waves[(t_end_waves > r) & (t_end_waves <= r + window + 300)].dropna()

        # Проверяем, все ли пики найдены
        if not (
                p_start.empty or p_peak.empty or p_end.empty or q.empty or s.empty or t_start.empty or t_peak.empty or t_end.empty):
            cycles.append({
                "P_start": p_start.iloc[0],
                "P_peak": p_peak.iloc[0],
                "P_end": p_end.iloc[0],
                "Q": q.iloc[0],
                "R": r,
                "S": s.iloc[0],
                "T_start": t_start.iloc[0],
                "T_peak": t_peak.iloc[0],
                "T_end": t_end.iloc[0],
            })

    """
    p_start_waves = p_start_waves[p_start_waves.first_valid_index():p_start_waves.last_valid_index() + 1]
    p_peaks = p_peaks[p_peaks.first_valid_index():p_peaks.last_valid_index() + 1]
    p_end_waves = p_end_waves[p_end_waves.first_valid_index():p_end_waves.last_valid_index() + 1]
    q_waves = q_waves[q_waves.first_valid_index():q_waves.last_valid_index() + 1]
    r_peaks = r_peaks[r_peaks.first_valid_index():r_peaks.last_valid_index() + 1]
    s_waves = s_waves[s_waves.first_valid_index():s_waves.last_valid_index() + 1]
    t_start_waves = t_start_waves[t_start_waves.first_valid_index():t_start_waves.last_valid_index() + 1]
    t_peaks = t_peaks[t_peaks.first_valid_index():t_peaks.last_valid_index() + 1]
    t_end_waves = t_end_waves[t_end_waves.first_valid_index():t_end_waves.last_valid_index() + 1]
    """

    """
    ###############################################################################
    # Нужно найти для каждого P соответствующий R позже него, и тогда всё будет ок.

    peak_q = 0
    minimal_p_start_wave = p_start_waves.iloc[0]

    for peak in q_waves:
        peak_q = peak
        if peak > minimal_p_start_wave:
            break


    q_waves = q_waves[q_waves >= peak_q]
    r_peaks = r_peaks[r_peaks >= peak_q]
    s_waves = s_waves[s_waves >= peak_q]

    min_length = min(len(r_peaks), len(p_start_waves))

    # Вирівняти обидва масиви по довжині
    p_start_waves = p_start_waves[:min_length] # Зріз до індексу min_length, не включаючи його
    p_peaks = p_peaks[:min_length]
    p_end_waves = p_end_waves[:min_length]
    r_peaks = r_peaks[:min_length]
    q_waves = q_waves[:min_length]
    s_waves = s_waves[:min_length]


    mask_t_start = set(np.where(~np.isnan(t_start_waves))[0])
    mask_t_peaks = set(np.where(~np.isnan(t_peaks))[0])
    mask_t_end = set(np.where(~np.isnan(t_end_waves))[0])

    mask3 = mask_t_start.intersection(mask_t_peaks)
    mask4 = mask3.intersection(mask_t_end)
    common_indices_sorted = sorted(mask4)


    p_start_waves = p_start_waves.reset_index(drop=True)
    p_end_waves = p_end_waves.reset_index(drop=True)
    q_waves = q_waves.reset_index(drop=True)
    r_peaks = r_peaks.reset_index(drop=True)
    s_waves = s_waves.reset_index(drop=True)
    t_start_waves = t_start_waves.reset_index(drop=True)
    t_end_waves = t_end_waves.reset_index(drop=True)

    p_start_waves = p_start_waves[common_indices_sorted]
    p_end_waves = p_end_waves[common_indices_sorted]
    q_waves = q_waves[common_indices_sorted]
    r_peaks = r_peaks[common_indices_sorted]
    s_waves = s_waves[common_indices_sorted]
    t_start_waves = t_start_waves[common_indices_sorted]
    t_peaks = t_peaks[common_indices_sorted]
    t_end_waves = t_end_waves[common_indices_sorted]
    

    ###############################################################################
    # Нужно найти для каждого R соответствующий T позже него, и тогда всё будет ок.

    minimal_r_start_wave = r_peaks.iloc[0]

    index_start_from = 0

    for t_wave in t_start_waves:
        if t_wave > minimal_r_start_wave:
            break
        index_start_from += 1

    index_peak_from = 0

    for t_wave in t_peaks:
        if t_wave > minimal_r_start_wave:
            break
        index_peak_from += 1

    index_end_from = 0

    for t_wave in t_end_waves:
        if t_wave > minimal_r_start_wave:
            break
        index_end_from += 1

    t_start_waves = t_start_waves[index_start_from:]
    t_peaks = t_peaks[index_peak_from:]
    t_end_waves  = t_end_waves[index_end_from:]

    min_length = min(len(r_peaks), len(t_start_waves))

    # Вирівняти обидва масиви по довжині
    r_peaks = r_peaks[:min_length]
    q_waves = q_waves[:min_length]
    s_waves = s_waves[:min_length]
    t_start_waves = t_start_waves[:min_length]  # Зріз до індексу min_length, не включаючи його
    t_peaks = t_peaks[:min_length]
    t_end_waves = t_end_waves[:min_length]
    p_start_waves = p_start_waves[:min_length]
    p_peaks = p_peaks[:min_length]
    p_end_waves = p_end_waves[:min_length]
    """
    ################################################################################
    # Можливо перевірити випадок, коли останні значення не співпадають

    print("P start waves: ", p_start_waves)
    print("P end waves: ", p_end_waves)
    print("Q waves: ", q_waves)
    print("R peaks: ", r_peaks)
    print("S waves: ", s_waves)
    print("T start waves: ", t_start_waves)
    print("T end waves: ", t_end_waves)

    """
    ################### Mask NaN values #######################
    mask = ~np.isnan(p_start_waves)
    mask = mask.reset_index(drop=True)



    p_start_waves = p_start_waves.reset_index(drop=True)
    p_end_waves = p_end_waves.reset_index(drop=True)
    q_waves = q_waves.reset_index(drop=True)
    r_peaks = r_peaks.reset_index(drop=True)
    s_waves = s_waves.reset_index(drop=True)
    t_start_waves = t_start_waves.reset_index(drop=True)
    t_end_waves = t_end_waves.reset_index(drop=True)


    p_start_waves = p_start_waves[mask]
    p_end_waves = p_end_waves[mask]
    q_waves = q_waves[mask]
    r_peaks = r_peaks[mask]
    s_waves = s_waves[mask]
    t_start_waves = t_start_waves[mask]
    t_end_waves = t_end_waves[mask]

    mask2 = ~np.isnan(p_end_waves)
    mask2 = mask2.reset_index(drop=True)

    p_start_waves = p_start_waves.reset_index(drop=True)
    p_end_waves = p_end_waves.reset_index(drop=True)
    q_waves = q_waves.reset_index(drop=True)
    r_peaks = r_peaks.reset_index(drop=True)
    s_waves = s_waves.reset_index(drop=True)
    t_start_waves = t_start_waves.reset_index(drop=True)
    t_end_waves = t_end_waves.reset_index(drop=True)

    p_start_waves = p_start_waves[mask2]
    p_end_waves = p_end_waves[mask2]
    q_waves = q_waves[mask2]
    r_peaks = r_peaks[mask2]
    s_waves = s_waves[mask2]
    t_start_waves = t_start_waves[mask2]
    t_end_waves = t_end_waves[mask2]
    ###########################################################
    """



    # Инициализируем словарь с пустыми списками для каждого типа пика
    waves = {
        "ECG_P_Onsets": [],
        "ECG_P_Peaks": [],
        "ECG_P_Offsets": [],
        "ECG_Q_Peaks": [],
        "ECG_R_Peaks": [],
        "ECG_S_Peaks": [],
        "ECG_T_Onsets": [],
        "ECG_T_Peaks": [],
        "ECG_T_Offsets": []
    }

    # Заполняем словарь данными из каждого цикла
    for cycle in cycles:
        waves["ECG_P_Onsets"].append(cycle["P_start"])
        waves["ECG_P_Peaks"].append(cycle["P_peak"])
        waves["ECG_P_Offsets"].append(cycle["P_end"])
        waves["ECG_Q_Peaks"].append(cycle["Q"])
        waves["ECG_R_Peaks"].append(cycle["R"])
        waves["ECG_S_Peaks"].append(cycle["S"])
        waves["ECG_T_Onsets"].append(cycle["T_start"])
        waves["ECG_T_Peaks"].append(cycle["T_peak"])
        waves["ECG_T_Offsets"].append(cycle["T_end"])

    # Преобразуем списки в pd.Series
    for key in waves:
        waves[key] = pd.Series(waves[key])

    """
    waves = {"ECG_P_Onsets": p_start_waves, "ECG_P_Peaks": p_peaks, "ECG_P_Offsets": p_end_waves,
             "ECG_Q_Peaks": q_waves,
             "ECG_R_Peaks": r_peaks, "ECG_S_Peaks": s_waves, "ECG_T_Onsets": t_start_waves, "ECG_T_Peaks": t_peaks,
             "ECG_T_Offsets": t_end_waves}
    """

    # Примерные данные
    time = np.linspace(0, 4000, len(cleaned_signal))  # Время в мс

    # ===== MAIN =====
    #record_name = 'sample-data/100'  # укажи свой путь к MIT-BIH записи
    #record = wfdb.rdrecord(record_name)
    #fs = record.fs
    #signal_data = record.p_signal[:, 0]  # первый канал

    # Фильтрация
    #filtered = bandpass_filter(signal_data, fs)

    # Поиск R-пиков
    #r_peaks = find_r_peaks(filtered, fs)

    # Анализ волны P
    #p_wave_ends = analyze_p_wave(cleaned_signal, 1000, waves["ECG_R_Peaks"])

    # ===== Визуализация =====
    """
    plt.figure(figsize=(12, 4))
    plt.plot(cleaned_signal, label="ECG")
    plt.plot(r_peaks, cleaned_signal[r_peaks], 'ro', label='R-peaks')
    for p in p_wave_ends:
        if p:
            plt.axvline(x=p, color='g', linestyle='--', alpha=0.6)
    plt.legend()
    plt.title("Конец P-волны (зеленые линии)")
    plt.xlabel("Samples")
    plt.show()
    """
    p_start_indices = waves["ECG_P_Onsets"]
    p_end_indices = waves["ECG_P_Offsets"]  # Индексы концов зубцов P
    q_start_indices = waves["ECG_Q_Peaks"]  # Индексы началов Q

    # Обчислення ізолінії
    # Disabled isoline
    isoline = estimate_isoline(cleaned_signal, p_end_indices, q_start_indices)
    #print(f"Оценённая изолиния: {isoline:.4f} мВ")
    cleaned_signal = cleaned_signal - isoline
    # !!! DISABLED S PEAKS detection
    #s_peaks = detect_s_peaks(waves["ECG_R_Peaks"], cleaned_signal)
    # Подменяем S-пики на свои:
    #waves["ECG_S_Peaks"] = pd.Series(np.array(s_peaks))
    # Визуализируем с кастомными S-пиками:


    """
    # Визуализация
    plt.plot(time, cleaned_signal, label='ECG')
    plt.axhline(y=isoline, color='gray', linestyle='--', label='Изолиния')
    plt.legend()
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.title('Изолиния на основе PQ-сегмента')
    plt.grid()
    plt.show()
    """
    #cleaned_signal = cleaned_signal - isoline

    ECG_PARAMETERS = {}

    p_duration = find_P_interval(waves["ECG_P_Onsets"], waves["ECG_P_Offsets"])    #!!!
    t_duration = find_T_interval(waves["ECG_T_Onsets"], waves["ECG_T_Offsets"])    #!!!

    ###############################################################################
    # Calculate heart contraction frequency - 1-st parameter (heart rate)
    HCF = calculate_HCF(waves["ECG_R_Peaks"])                        #!!!
    print("Heart rate: ",HCF)
    ECG_PARAMETERS["Heart rate"] = HCF

    ################################ Перевіряємо, чи PR інтервали однакові ###########################
    corrected_pq_intervals = corrected_PQ_intervals(waves["ECG_Q_Peaks"], waves["ECG_P_Onsets"])

    # Перевіримо варіацію:
    std_dev = np.std(corrected_pq_intervals)
    mean_val = np.mean(corrected_pq_intervals)
    coefficient_of_variation = std_dev / mean_val       #!!!

    print(f"PQ intervals: {corrected_pq_intervals}")

    # 2-nd and 3-rd parameters (mean PR intervals, CoefVar)
    print(f"Mean PQ: {mean_val:.2f}, Std: {std_dev:.2f}, CoefVar: {coefficient_of_variation:.4f}")
    ECG_PARAMETERS["PQ"] = mean_val
    ECG_PARAMETERS["PQ CoefVar"] = coefficient_of_variation
    ##################################################################################################

    # 4-th parameter (mean RR intervals)
    mean_RR = mean_RR_interval(waves["ECG_R_Peaks"])

    print(f"Mean RR: ", mean_RR)
    ECG_PARAMETERS["RR"] = mean_RR

    # 5-th parameter (mean ST interval)
    mean_ST = mean_ST_interval(waves["ECG_S_Peaks"], waves["ECG_T_Onsets"])

    print(f"Mean ST segment: ", mean_ST)
    ECG_PARAMETERS["ST"] = mean_ST

    # 6-th parameter (QRS complex)
    mean_QRS = mean_QRS_complex(waves["ECG_Q_Peaks"], waves["ECG_S_Peaks"])

    print("Mean QRS: ", mean_QRS)
    ECG_PARAMETERS["QRS"] = mean_QRS
    # p_end = delineate_info["ECG_P_Offsets"][index]
    # q_start = delineate_info["ECG_Q_Peaks"][index]
    # s_end = delineate_info["ECG_S_Peaks"][index]
    # t_start = delineate_info["ECG_T_Onsets"][index]


    # 7-th and 8-th parameters (P duration, T duration)
    print("P duration: ", p_duration)
    ECG_PARAMETERS["P"] = p_duration
    print("T duration: ", t_duration)
    ECG_PARAMETERS["T"] = t_duration

    print(p_peaks.min(), p_peaks.max(), len(cleaned_signal))

    #9-th, 10-th and 11-th parameter
    p_amplitude = np.mean(cleaned_signal[waves["ECG_P_Peaks"].to_numpy().astype(int)])
    r_amplitude = np.mean(cleaned_signal[waves["ECG_R_Peaks"].to_numpy().astype(int)])
    t_amplitude = np.mean(cleaned_signal[waves["ECG_T_Peaks"].to_numpy().astype(int)])

    print("P amplitude: ", p_amplitude)
    print("R amplitude: ", r_amplitude)
    print("T amplitude: ", t_amplitude)

    ECG_PARAMETERS["P amplitude"] = p_amplitude
    ECG_PARAMETERS["R amplitude"] = r_amplitude
    ECG_PARAMETERS["T amplitude"] = t_amplitude
    # print("P end ",p_end) #Index of P end
    # print(q_start)  #Index of q start
    # PQ сегмент:
    # pq_segment = cleaned_signal[p_end:q_start]

    # ST сегмент:
    # st_segment = cleaned_signal[s_end:t_start]
    # print(pq_segment)
    # print(st_segment)

    """
    print(waves_peaks['ECG_T_Peaks'])
    t_peaks = np.array(waves_peaks['ECG_T_Peaks'])
    t_peaks = t_peaks[~np.isnan(t_peaks)].astype(int)

    time = np.linspace(0, len(ecg_signal) / fs, len(ecg_signal))
    """

    # Обчислюємо PR-інтервали
    #pr_intervals = np.array(r_peaks[1:len(r_peaks) - 1]) - waves_peaks["ECG_P_Onsets"][1:len(r_peaks) - 1]

    #for i in range(len(p_start_waves)):
     #   print(f"P: {p_start_waves[i]}, R: {r_peaks[i]}, PR: {r_peaks[i] - p_start_waves[i]}")




    #print(r_peaks[1:len(r_peaks) - 1])
    #print(p_start_waves)
    # Видаляємо NaN
    #pr_intervals = pr_intervals[~np.isnan(pr_intervals)]
    return isoline, waves, ECG_PARAMETERS

    #return waves, ECG_PARAMETERS


def find_P_interval(p_start_waves, p_end_waves):
    # Отримання початкової та кінцевої точок P-інтервалу для всіх серцевих циклів,
    # окрім першого та останнього
    """p_start_waves - список початків p піків
       P_end_waves - список кінців p піків"""

    p_start_waves = np.array(p_start_waves)
    p_end_waves = np.array(p_end_waves)

    p_diff_list = p_end_waves - p_start_waves

    # Видаляємо NaN
    p_diff_list = p_diff_list[~np.isnan(p_diff_list)]
    # Середня тривалість P-інтервалу
    p_duration = mean(p_diff_list)

    return p_duration

def find_T_interval(t_start_waves, t_end_waves):
    # Отримання початкової та кінцевої точок T-інтервалу для всіх серцевих циклів,
    # окрім першого та останнього

    t_start_waves = np.array(t_start_waves)
    t_end_waves = np.array(t_end_waves)

    t_diff_list = t_end_waves - t_start_waves

    # Видаляємо NaN
    t_diff_list = t_diff_list[~np.isnan(t_diff_list)]
    # Середня тривалість T-інтервалу
    t_duration = mean(t_diff_list)

    return t_duration

def plot_ECG_features(cleaned_signal, waves_peaks, count_plot):
    """Plot ECG features function

        input:
            cleaned_signal - filtered signal
            waves_peaks - peaks of waves
            count_plot - points to plot"""

    # Changable
    signal = cleaned_signal[:480000]
    x = np.arange(len(signal))

    ############################################## Plot peaks #########################################################

    plt.figure(figsize=(12, 6))

    #Plot filtered signal
    plt.plot(x, signal, label="ECG", color="black")

    #Plot R-peaks
    ecg_R_peaks = waves_peaks["ECG_R_Peaks"][:count_plot]
    plt.scatter(x[ecg_R_peaks], signal[ecg_R_peaks], color='red', label="R-peaks")

    #Plot P-peaks
    #print(waves_peaks["ECG_P_Peaks"][:count_plot])
    ecg_P_peaks = np.array(waves_peaks["ECG_P_Peaks"][:count_plot]).astype(int)
    #ecg_P_peaks = waves_peaks["ECG_P_Peaks"][:count_plot]
    plt.scatter(x[ecg_P_peaks], signal[ecg_P_peaks], color='green', label="P-peaks")

    #Plot T-peaks
    ecg_T_peaks = waves_peaks["ECG_T_Peaks"].to_numpy().astype(int)[:count_plot]
    #ecg_T_peaks = waves_peaks["ECG_T_Peaks"][:count_plot]
    plt.scatter(x[ecg_T_peaks], signal[ecg_T_peaks], color='blue', label="T-peaks")

    # Изолиния
    plt.axhline(y=0, color="gray", linestyle="--", linewidth=1, label="Isoline")


    # Custom events (insert your lists)
    def mark_events(event_indices, color, label, style='--'):
        for i in event_indices:
            plt.axvline(x=i, color=color, linestyle=style, linewidth=1.5)
        # Add only once in legend
        if len(event_indices) > 0:
            plt.axvline(x=event_indices.iloc[0], color=color, linestyle=style, label=label, linewidth=1.5)

    # Пример: замените списки на ваши
    mark_events(waves_peaks["ECG_P_Onsets"][:count_plot], "green", "P start")
    mark_events(waves_peaks['ECG_P_Peaks'][:count_plot], "lime", "P peak", style='-.')
    mark_events(waves_peaks["ECG_P_Offsets"][:count_plot], "green", "P end", style=':')

    mark_events(waves_peaks['ECG_Q_Peaks'][:count_plot], "blue", "Q", style='--')
    mark_events(waves_peaks['ECG_R_Peaks'][:count_plot], "red", "R peak", style='--')
    mark_events(waves_peaks['ECG_S_Peaks'][:count_plot], "blue", "S", style='--')

    mark_events(waves_peaks["ECG_T_Onsets"][:count_plot], "purple", "T start")
    mark_events(waves_peaks['ECG_T_Peaks'][:count_plot], "magenta", "T peak", style='-.')
    mark_events(waves_peaks["ECG_T_Offsets"][:count_plot], "purple", "T end", style=':')

    # Легенда и стили
    plt.legend(loc='upper right')
    plt.title("Кастомная визуализация зубцов ЭКГ")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def corrected_PQ_intervals(q_peaks, p_start_waves):
    # Нужно найти для каждого P соответствующий R позже него, и тогда всё будет ок.

    corrected_pq_intervals = []

    for p in p_start_waves:
        q_candidates = q_peaks[q_peaks > p]
        if len(q_candidates) > 0:
            nearest_q = q_candidates.iloc[0]
            corrected_pq_intervals.append(nearest_q - p)

    corrected_pq_intervals = np.array(corrected_pq_intervals)
    return corrected_pq_intervals


def extract_from_files_ids(files):
    """Extract from files RR intervals time series

        input: files - file names
        output: rr_time_series_dictionary - dictionary with id as key and list of RR-intervals as value"""

    import re

    list_of_ids = []

    for file in files:
        filename = file

        # Используем регулярное выражение для извлечения числового индекса
        match = re.search(r'_(\d+)\.txt', filename)
        if match:
            index = match.group(1)
            list_of_ids.append(index)

    return list_of_ids

def list_files_with_rr_intervals():
    """Get list of files with rr_intervals time series from rr_interval/all folder"""
    import os

    directory = rr_intervals_folder

    # Фильтрация только файлов
    files = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]

    for file in files:
        print(file)

    return files

def write_ECG_parameters_to_csv(sex, id, age_range, features):
    # ECG 1 and 2 simulationusly

    import csv
    import os

    filename = 'output/{0}_ECGs_features_calculated.csv'.format(sex)
    file_exists = os.path.exists(filename)
    file_empty = not file_exists or os.stat(filename).st_size == 0

    with open('output/{0}_ECGs_features_calculated.csv'.format(sex), 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # Додаємо заголовок
        if file_empty:
            spamwriter.writerow([
                "ID", "Age Group", "Heart rate", "PQ", "PQ CoefVar",
                "RR", "ST", "QRS", "P", "T", "P amplitude",
                "R amplitude", "T amplitude"
            ])


        spamwriter.writerow([id, age_groups[age_range],
                            features["Heart rate"], features["PQ"], features["PQ CoefVar"],
                            features["RR"], features["ST"], features["QRS"],
                            features["P"], features["T"], features["P amplitude"],
                            features["R amplitude"], features["T amplitude"]])





read_ECGs_annotation_data(False, True)


