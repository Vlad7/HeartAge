import numpy as np
import os
import re
import matplotlib.pyplot as plt

######################################################################################################
################################### LOAD RR-INTERVALS TIME SERIES ####################################
######################################################################################################
def get_list_of_files_with_rr_intervals(rr_intervals_folder):
    """Get list of files with rr_intervals time_series from rr_intervals_folder"""

    # Фільтрація лише файлів только файлов
    files = [file for file in os.listdir(rr_intervals_folder) if os.path.isfile(os.path.join(rr_intervals_folder, file))]

    for file in files:
        print(file)

    return files

def extract_from_files_rr_intervals_time_series(rr_intervals_folder, files):
    """Extract from files RR intervals time series

        input: files - file names
        output: rr_time_series_dictionary - dictionary with id as key and list of RR-intervals as value"""

    rr_time_series_dictionary = {}

    for filename in files:

        index, rr_time_series = extract_from_file_rr_intervals_time_series(rr_intervals_folder, filename)
        if index != None:
            rr_time_series_dictionary[index] = rr_time_series

    return rr_time_series_dictionary

def extract_from_file_rr_intervals_time_series(rr_intervals_folder, filename):
    """Extract from file RR intervals time series

        input: file - file name
        output: rr_intervals_time_series - list of RR-intervals"""

    rr_intervals_time_series = []

    # Используем регулярное выражение для извлечения числового индекса
    match = re.search(r'_(\d+)\.txt', filename)
    if match:
        index = match.group(1)
        # print("Индекс:", index)

        file_path = rr_intervals_folder + "/" + filename
        # Чтение файла, начиная со второй строки
        with open(file_path, "r") as file:
            # Пропускаем первую строку
            next(file)

            # Читаем остальные строки
            rr_intervals_time_series = [line.strip() for line in file]

        rr_intervals_time_series = [int(float(x)) for x in rr_intervals_time_series]

        return index, rr_intervals_time_series

    else:
        print("Index doesn't found")
        return None, None



######################################################################################################
################################### RR-INTERVALS TIME SERIES INFORMATION #############################
######################################################################################################


def find_minimum_rr_count(rr_time_series_dictionary):
    """Find minimum count of rr-intervals in lists of dictionary values"""
    # Find minimum len of rr_time_series
    min_len = 1000000
    min_id = None
    for id in rr_time_series_dictionary.keys():
        ln = len(rr_time_series_dictionary[id])
        if ln < min_len:
            min_len = ln
            min_id = id

    return min_id, min_len

def check_for_minimum_time_rr_time_series(rr_time_series_dictionary, min_time=300000):
    """Check, if summ of RR intervals of time series less than 5 min"""

    for key in rr_time_series_dictionary.keys():
        summ = np.sum(rr_time_series_dictionary[key])
        if summ < min_time:
            print("Record of time series less than 5 minutes!")
            return False

    return True

######################################################################################################
################################### RR-INTERVALS TIME SERIES CUTTING #################################
######################################################################################################

def preprocess_length_of_rr_intervals_time_series(rr_intervals_time_series, mode="fixed_count", count=440, duration=300000):
    """Preprocessing of RR-intervals: selecting fixed number of points or time interval. Maybe for future add different
    methods for cut.

    :param rr_intervals_time_series: массив RR-интервалів (в мс)
    :param mode: "fixed_count" (фіксована кількість) или "fixed_duration" (фіксована тривалість)
    :param count: кількість RR-интервалів (наприклад, 500)
    :param duration: тривалість аналізу в мс (наприклад, 300000 мс = 5 хвилин)
    :return: опрацьований масив RR-интервалів
    """
    rr_intervals_time_series = np.array(rr_intervals_time_series)  # Перетворюємо в массив numpy
    avg_rr = np.mean(rr_intervals_time_series)                     # Середній RR-інтервал
    hr = 60000 / avg_rr                                            # ЧСС (уд/хв)

    print(f"Середній RR-інтервал: {avg_rr:.2f} мс, ЧСС: {hr:.2f} уд/мин")

    if mode == "fixed_count":
        print("Всього: "+str(len(rr_intervals_time_series)))
        print(f"Обрано {count} RR-интервалів")
        return rr_intervals_time_series[:count]  # Беремо перші count точок

    elif mode == "fixed_duration":
        total_time = np.cumsum(rr_intervals_time_series)        # Сумуємо RR-інтервали
        valid_indices = np.where(total_time <= duration)[0]     # Шукаємо точки, що вкладаються в duration. Приклад
                                                                # a = np.array([10, 5, 20])
                                                                # np.where(a > 8)
                                                                # Поверне:
                                                                # (array([0, 2]),)
                                                                # Це кортеж довжини 1, всередині якого один масив індексів.
                                                                #
                                                                # Тобто.:
                                                                #
                                                                # np.where(...) → (array([...]),)
                                                                #
                                                                # [0] → бере перший елемент кортежу → сам масив індексів
        print(f"Обрано {len(valid_indices)} RR-интервалів (на {duration / 1000} секунд)")
        return rr_intervals_time_series[valid_indices]  # Возвращаем только эти точки

    else:
        raise ValueError("Неправильний режим. Використовуйте 'fixed_count' або 'fixed_duration'.")


def plot_RR_intervals_time_series(rr_intervals, first_time=40000, second_time=54000):
    """Plot RR intervals time series in the time range"""
    # Extracting RR intervals

    print(rr_intervals)

    # Создаём массив накопленного времени
    cumulative_time = np.cumsum(rr_intervals)  # массив накопленного времени (в мс)

    print(cumulative_time)

    # Определяем индексы RR-интервалов в диапазоне 40-52 секунды (40000-52000 мс)
    start_idx = np.searchsorted(cumulative_time, first_time)  # первый индекс
    end_idx = np.searchsorted(cumulative_time, second_time)  # последний индекс
    # Отбираем данные для построения графика
    filtered_rr = rr_intervals[start_idx:end_idx]
    filtered_time = cumulative_time[start_idx:end_idx]
    # print(filtered_time)

    # Создаём ось X (по номеру R-R)
    filtered_numbers = list(range(start_idx + 1, end_idx + 1)) # + 1 поскольку натуральные числа, нумеруется от 1
    print(filtered_numbers)
    # Для графика создадим массив с метками времени, который соответствует каждому интервалу
    # Так как частота дискретизации 1000 Гц, то временная ось будет с шагом 1 мс
    sampling_rate = 1000
    print(len(rr_intervals))
    # time_axis = [i / sampling_rate for i in range(len(rr_intervals))]  # временные метки с шагом 1 мс
    #number_axis = [i + 1 for i in range(len(rr_intervals))]
    print(filtered_numbers)
    # print(number_axis)
    # Строим график
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_numbers, filtered_rr, marker='o', color='b', linestyle='-', label='RR-intervals')
    plt.title('RR intervals')
    plt.xlabel('N (R-R)')
    plt.ylabel('RR-interval (ms)')
    # Устанавливаем метки на оси X через 1
    plt.xticks(filtered_numbers)  # Устанавливаем все числа в качестве подписей
    plt.grid(True)
    plt.legend()
    plt.show()