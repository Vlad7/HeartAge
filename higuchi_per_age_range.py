import numpy as np
import HiguchiFractalDimension.hfd
import main2 as m2
import re
import os
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
from scipy import stats


#Folder with files in each there is cleaned_signal time series
<<<<<<< HEAD
cleaned_signal_folder="../dataset/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0/cleaned_signal"
=======
cleaned_signal_folder="../datasets/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0/cleaned_signal"
>>>>>>> 1432ed2d6a436909ae1c7e552f7104969b943b81

#Folder with files in each there is r_peaks time series
r_peaks_folder="r_peaks"

def list_files_with_cleaned_signal():
    """Get list of files with cleaned_signal time series from cleaned_signal"""
    import os

    directory = cleaned_signal_folder

    # Фильтрация только файлов
    files = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]

    for file in files:
        print(file)

    return files

def list_files_with_r_peaks():
    """Get list of files with cleaned_signal time series from cleaned_signal"""
    import os

    directory = r_peaks_folder

    # Фильтрация только файлов
    files = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]

    for file in files:
        print(file)

    return files


def extract_from_file_cleaned_signal_time_series(file):
    """Extract from files RR intervals time series

        input: files - file names
        output: rr_time_series_dictionary - dictionary with id as key and list of RR-intervals as value"""


    filename = file

    # Используем регулярное выражение для извлечения числового индекса
    match = re.search(r'_(\d+)\.txt', filename)
    if match:
        index = match.group(1)

        #print("Индекс:", index)

        file_path = cleaned_signal_folder + "/" + file
        # Чтение файла, начиная со второй строки
        with open(file_path, "r") as file:
            # Пропускаем первую строку
            next(file)

            # Читаем остальные строки
            cleaned_signal = [line.strip() for line in file]

        # Вывод значений
        #for rr in rr_intervals:
        #    print(rr)

        cleaned_signal = [float(x) for x in cleaned_signal]
        #print(rr_intervals)
        return index, cleaned_signal

    else:
        print("Индекс не найден")
        return None, None



def extract_from_file_r_peaks_time_series(cleaned_signal_file, r_peaks_file):
    """Extract from files RR intervals time series

        input: files - file names
        output: rr_time_series_dictionary - dictionary with id as key and list of RR-intervals as value"""

    import re

    filename = cleaned_signal_file

    # Используем регулярное выражение для извлечения числового индекса
    match = re.search(r'_(\d+)\.txt', filename)
    if match:
        index = match.group(1)

        if index in r_peaks_file:
            #print("Индекс:", index)

            file_path = r_peaks_folder + "/" + r_peaks_file
            # Чтение файла, начиная со второй строки
            with open(file_path, "r") as file:
                # Пропускаем первую строку
                next(file)

                # Читаем остальные строки
                r_peaks = [line.strip() for line in file]

        # Вывод значений
        #for rr in rr_intervals:
        #    print(rr)

                r_peaks = [int(float(x)) for x in r_peaks]
                #print(rr_intervals)
                return r_peaks

        else:
            print("Индекс не найден")
            return None



def zscore_normalize(x):
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma == 0:
        return x - mu   # если плоский сигнал
    return (x - mu) / sigma



def higuchi_fd(seg, kmax):
    #HFD = HiguchiFractalDimension.hfd(seg, opt=True,
    #                                    k_max=
    k, L = HiguchiFractalDimension.curve_length(seg, opt=True, k_max=kmax)


    # строим регрессию log–log
    x = np.log2(k)
    y = np.log2(L)

    res = stats.linregress(x, y)

    k = res.slope
    b = res.intercept
    r2 = res.rvalue**2
    p = res.pvalue

    print(f"Slope = {k:.4f}")
    print(f"Intercept = {b:.4f}")
    print(f"R^2 = {r2:.4f}")
    print(f"p-value for slope = {p:.4e}")

    hfd = -k
    print(f"Higuchi fractal dimension D = {hfd:.4f}")




    # предсказанные значения по прямой
    y_pred = k  * x + b











    # Твои данные
    # k = np.array([...])
    # L = np.array([...])

    # Визуализация
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, label="Data (log2)", color="blue")
    plt.plot(x, y_pred, label=f"Fit: y = {k:.2f}x + {b:.2f}\nD = {hfd:.3f}, R² = {r2:.3f}, \np-value = {p}",  color="red")
    plt.xlabel("log2(k)")
    plt.ylabel("log2(L(k))")
    plt.title("Higuchi Fractal Dimension (log-log)")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.6)
    plt.show()

    return [k, b, hfd, r2, p]




def windowed_hfd_cycles(x: np.ndarray, rpeaks_idx: np.ndarray, n_cycles: int = 100, step_cycles: int = 20, kmax: int = 10):
    """
    Оконный HFD по фиксированному числу сердечных циклов.
    rpeaks_idx: индексы R-пиков в отсчетах (возрастающий массив)
    n_cycles: сколько последовательных комплексов берём в окно
    step_cycles: шаг по циклам
    Возвращает: centers_idx (индексы центра окна, можно перевести в секунды), hfd_values
    """


    rpeaks_idx = np.asarray(rpeaks_idx, dtype=int)
    if rpeaks_idx.size < n_cycles + 1:
        return np.array([]), np.array([])

    # формируем пары границ по R-пикам: [R_i, R_{i+n_cycles}]
    starts = np.arange(0, rpeaks_idx.size - n_cycles - 0, step_cycles) #!!! 180 max
    hfd_vals = []
    centers = []
    #print(starts)
    for i in starts:
        #print(rpeaks_idx)
        a = rpeaks_idx[i]
        b = rpeaks_idx[i + n_cycles]
        #print("a: {0}, b: {1}".format(a,b))
        if b - a < 2:
            #print("less than 2")
            hfd_vals.append(np.nan)
            centers.append((a + b) // 2)
            continue
        seg = x[a:b]
        #print("Длина сегмента: "+str(len(seg)))
        hfd_vals.append(higuchi_fd(seg, kmax=kmax))
        centers.append((a + b) // 2)
    return np.array(centers), np.array(hfd_vals)

def write_average_HFD_values_for_each_age_range(sex, higuchi_average_per_each_age_group):
    with open('output/{0}_HFD_average_of_ECG_per_age_range.csv'.format(sex), 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for age_group in higuchi_average_per_each_age_group.keys():
            spamwriter.writerow([m2.age_groups[age_group], f"{higuchi_average_per_each_age_group[age_group]:.3f}".replace('.', ',')])

def write_HFD_calculated_value_to_csv(sex, id, info):
    # ECG 1 and 2 simulationusly

    file_path = 'output/{0}_HFD_all_ECG_calculated_separeted_windows.csv'.format(sex)

    # Проверяем, существует ли файл и пуст ли он
    file_exists = os.path.isfile(file_path)
    file_empty = not file_exists or os.path.getsize(file_path) == 0

    with open(file_path, 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        windows_count = len(info)
        # Если файл пустой, пишем заголовок
        if file_empty:

            list = ['id']
            for i in range(1, windows_count, 1):
                list += ['k{0}'.format(i), 'b{0}'.format(i), 'D{0}'.format(i), 'R_score{0}'.format(i), 'p-value{0}'.format(i)]
            spamwriter.writerow(list)

        # Добавляем новые строки


        list = [id]
        for i in range(0, windows_count, 1):
            # info[i][0] - i-th window k parameter
            # info[i][1] - i-th window b parameter
            # info[i][2] - i-th window D parameter
            # info[i][3] - i-th window R_square parameter
            list += [f"{info[i][0]:.3f}", f"{info[i][1]:.3f}", f"{info[i][2]:.3f}", f"{info[i][3]:.3f}", f"{info[i][4]:.3f}"]
        spamwriter.writerow(list)





        """, RECORD.DATABASE[type_of_ecg_cut][key].Sex,
                                 RECORD.DATABASE[type_of_ecg_cut][key].BMI]"""

def create_id_to_hfd_file():
    cleaned_signals_files = list_files_with_cleaned_signal()
    r_peaks_files = list_files_with_r_peaks()

    if len(cleaned_signals_files) != len(r_peaks_files):
        print("ERROR!")

    # Dictionary with HFD for each id
    # HFD = {}

    for cleaned_signal_file, r_peaks_file in zip(cleaned_signals_files, r_peaks_files):
        id, cleaned_time_series = extract_from_file_cleaned_signal_time_series(cleaned_signal_file)
        peaks = extract_from_file_r_peaks_time_series(cleaned_signal_file, r_peaks_file)

        normalized = zscore_normalize(cleaned_time_series)

        #  Видалити значення менші нуля на початку peaks
        while peaks and peaks[0] < 0:
            peaks.pop(0)

        min_length = 480501
        cycle_duration = 1500
        min_cycles = min_length / cycle_duration
        get_cycles = int(min_cycles / 100) * 100
        selected_R_peak_cycles = peaks[:get_cycles]
        print(cleaned_signal_file)
        print(selected_R_peak_cycles)
        print("Cycles: " + str(get_cycles))

        print("Length: " + str(len(normalized)))
        print("Peaks: " + str(len(selected_R_peak_cycles)))
        # Тут закінчив.

        #Firstly kmax=1500, step_cycles = 20
        centers_idx, info = windowed_hfd_cycles(normalized, selected_R_peak_cycles, n_cycles=100, step_cycles=50,
                                               kmax=10000)

        #info - list with lists with [k, b, D, R_score]
        #hfd_mean = np.mean(hfd)
        #print(hfd_mean)

        match = re.search(r'_(\d+)\.txt', cleaned_signal_file)
        id = match.group(1)

        #write_HFD_calculated_value_to_csv("both sexes", id, hfd_mean)
        write_HFD_calculated_value_to_csv("both sexes", id, info)
        # HFD[id] = hfd_mean
        # Для времени: t_centers = centers_idx / fs

        if peaks == None:
            break
"""ВНИМАНИЕ! R-пики могут быть отрицательны"""

def load_id_to_hfd():

    id_to_hfd = {}

    with open('output/both_sexes_HFD_all_ECG_calculated.csv', newline='') as csvfile:
        spamreader = csv.DictReader(csvfile, delimiter=';', quotechar='|')
        for row in spamreader:
            id_ = f"{int(row['id']):04d}"
            hfd_value = float(row['HFD'])  # если числа с запятой
            id_to_hfd[id_] = hfd_value
    return id_to_hfd

def age_range_agregation (id_to_hfd, id_ageRangeIndex_dict):
    """Create {'age_range': mean_hfd'} dictionary"""

    print(id_ageRangeIndex_dict)

    # Сначала группируем HFD по age_range
    age_to_hfds = defaultdict(list)
    for id_, age_range in id_ageRangeIndex_dict.items():
        if id_ in id_to_hfd:  # проверка, чтобы id существовал в HFD
            age_to_hfds[age_range].append(id_to_hfd[id_])

    # Затем считаем среднее для каждого age_range
    age_to_mean_hfd = {age: float(np.mean(hfds)) for age, hfds in age_to_hfds.items()}

    print(age_to_mean_hfd)

    return age_to_mean_hfd

def age_range_agregation_count (id_to_hfd, id_ageRangeIndex_dict):
    """Create {'age_range': mean_hfd'} dictionary"""

    print(id_ageRangeIndex_dict)

    # Сначала группируем HFD по age_range
    age_to_hfds = defaultdict(list)
    for id_, age_range in id_ageRangeIndex_dict.items():
        if id_ in id_to_hfd:  # проверка, чтобы id существовал в HFD
            age_to_hfds[age_range].append(id_to_hfd[id_])

    # Затем считаем среднее для каждого age_range
    age_to_count = {age: len(hfds) for age, hfds in age_to_hfds.items()}

    print(age_to_count)

    return age_to_count

if __name__ == '__main__':

    create_id_to_hfd_file()
    """
    id_to_hfd = load_id_to_hfd()

    print(id_to_hfd)

    keys = id_to_hfd.keys()
    male_ids, female_ids = m2.classify_ids_by_sex(id_to_hfd.keys())

    print(male_ids)
    print(female_ids)

    male_id_ageRangeIndex_dict, female_id_ageRangeIndex_dict = m2.get_age_ranges_for_male_and_female(keys, male_ids, female_ids)

    #ale_age_range_to_mean_hfd = age_range_agregation(id_to_hfd, male_id_ageRangeIndex_dict)
    #female_age_range_to_mean_hfd = age_range_agregation(id_to_hfd, female_id_ageRangeIndex_dict)

    male_age_range_to_count = age_range_agregation_count(id_to_hfd, male_id_ageRangeIndex_dict)
    female_age_range_to_count = age_range_agregation_count(id_to_hfd, female_id_ageRangeIndex_dict)
    m2.write_number_of_ECGs_per_age_range_for_both_HFD("male", male_age_range_to_count)
    m2.write_number_of_ECGs_per_age_range_for_both_HFD("female", female_age_range_to_count)

    #write_average_HFD_values_for_each_age_range("male", male_age_range_to_mean_hfd)
    #write_average_HFD_values_for_each_age_range("female", female_age_range_to_mean_hfd)
    #write_average_HFD_values_for_each_age_range("both_sexes", age_to_mean_hfd)
    """



