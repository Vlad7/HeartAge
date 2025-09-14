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
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_reset

#Folder with files in each there is cleaned_signal time series

cleaned_signal_folder="../dataset/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0/cleaned_signal"


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



def extract_from_file_r_peaks_time_series(id, r_peaks_file):
    """Extract from file R peaks time series

        input: id - corresponding to clean and r_peaks
        output: r_peaks - list with R-peaks as value"""



    if id in r_peaks_file:
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



def higuchi_fd(seg, id, window, num_k, kmax):
    #HFD = HiguchiFractalDimension.hfd(seg, opt=True,
    #                                    k_max=
    is_plot = False
    k, L = HiguchiFractalDimension.curve_length(seg, opt=True, num_k=num_k, k_max=kmax)


    # строим регрессию log–log
    x = np.log2(k)
    y = np.log2(L)

    y_pred_linear, k, b, hfd, p_linear, lin_model_rsquared, lin_model_aic = linear_regression(x,y,is_plot)
    y_pred_quadro, kefs, p_squared, quadr_model_rsquared, quadr_model_aic = quadratic_regression(x,y,is_plot)


    print("\nСравнение моделей:")
    print("R^2 линейной:",lin_model_rsquared)
    print("R^2 квадратичной:", quadr_model_rsquared)
    print("AIC линейной:", lin_model_aic)
    print("AIC квадратичной:", quadr_model_aic)

    if is_plot:
        # Визуализация
        plt.figure(figsize=(7, 5))
        plt.scatter(x, y, label="Data (log2)", color="blue")
        plt.plot(x, y_pred_linear, label=f"Fit: y = {k:.2f}x + {b:.2f}\nD = {hfd:.3f}, R² = {lin_model_rsquared:.3f}, \np-value linear = {p_linear:.5f}, \nAIC linear = {lin_model_aic:.3f}", color="red")
        plt.plot(x, y_pred_quadro, label=f"Fit: y = {kefs[2]:.2f}x^2 + {kefs[1]:.2f}x + {kefs[0]:.2f},\nR² = {quadr_model_rsquared:.3f}, \np-value quadratic = {p_squared:.5f},\nAIC quadratic = {quadr_model_aic:.3f}",
                 color="green")

        plt.xlabel("log2(k)")
        plt.ylabel("log2(L(k))")
        plt.title("Higuchi Fractal Dimension (log-log) Id = {0}, window = {1}".format(id, window))
        plt.legend()
        plt.grid(True, ls="--", alpha=0.6)
        plt.show()

    return [k, b, hfd, p_linear, lin_model_rsquared, lin_model_aic, kefs[2], kefs[1], kefs[0], p_squared, quadr_model_rsquared, quadr_model_aic]


def linear_regression(x, y, is_plot):
    res = stats.linregress(x, y)

    ########### Строим линейную модель ###########

    X = sm.add_constant(x)  # добавляем константу
    model_lin = sm.OLS(y, X).fit()
    y_pred = model_lin.predict()
    residuals = y-y_pred

    # y_pred = res.intercept + res.slope * x

    # коэффициенты
    b = model_lin.params[0]  # beta_0
    k = model_lin.params[1]  # beta_1
    hfd = -k

    # p-value для каждого коэффициента
    p_value = model_lin.f_pvalue
    #p_intercept = p_values[0]
    #p_slope = p_values[1]



    # RESET тест (по умолчанию квадратичные и кубические термины)
    reset_test = linear_reset(model_lin, power=2, use_f=True)
    print("RESET-тест линейная модель:", reset_test)
    p_value_reset = reset_test.pvalue
    if p_value_reset > 0.0005:
        with open("result.txt", "w", encoding="utf-8") as f:
            f.write("NOT OK")

    if is_plot:
        plt.scatter(x, residuals, color="blue")
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel("log2(k)")
        plt.ylabel("Residuals")
        plt.title("Residuals linear vs log2(k)")
        plt.show()

    return y_pred, k, b, hfd, p_value, model_lin.rsquared, model_lin.aic

def quadratic_regression(x, y, is_plot):

    #coeffs = np.polyfit(x, y, deg=2)
    #y_quadro_predicted = np.polyval(coeffs, x)
    #y_quadro_predicted = coeffs[0] *x * x + coeffs[1]*x + coeffs[2]





    ########### квадратичная модель
    X_quad = sm.add_constant(np.column_stack([x, x ** 2]))
    model_quad = sm.OLS(y, X_quad).fit()
    y_quad_pred = model_quad.predict(X_quad)
    residuals_quad = y - y_quad_pred
    quadr_model_rsquared = model_quad.rsquared
    quadr_model_aic = model_quad.aic
    p_value=model_quad.f_pvalue
    # коэффициенты
    kefs = [model_quad.params[0], model_quad.params[1], model_quad.params[2]]  # b, ax1, cx2
    p_x2 = model_quad.pvalues[2]
    print("p-value для коэффициента при x^2:", p_x2)
    # считаем R^2

    #from sklearn.metrics import r2_score
    #r2_squared = r2_score(y, y_quadro_predicted)

    # Твои данные
    # k = np.array([...])
    # L = np.array([...])


    # R^2
    if is_plot:
        plt.scatter(x, residuals_quad, color="blue")
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel("log2(k)")
        plt.ylabel("Residuals")
        plt.title("Residuals quadratic vs log2(k)")
        plt.show()

    return y_quad_pred, kefs, p_value, quadr_model_rsquared, quadr_model_aic




def windowed_hfd_cycles(x: np.ndarray, rpeaks_idx: np.ndarray, id, num_k : int = 50, n_cycles: int = 100, step_cycles: int = 20,  kmax: int = 10):
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
    info_vals = []
    centers = []
    #print(starts)
    for i, w in zip(starts, range(0, len(starts),1)):
        #print(rpeaks_idx)
        a = rpeaks_idx[i]
        b = rpeaks_idx[i + n_cycles]
        #print("a: {0}, b: {1}".format(a,b))
        if b - a < 2:
            #print("less than 2")
            info_vals.append([np.nan, np.nan, np.nan, np.nan, np.nan])
            centers.append((a + b) // 2)
            continue
        seg = x[a:b]
        #print("Длина сегмента: "+str(len(seg)))
        info_vals.append(higuchi_fd(seg, id, w + 1, num_k, kmax=kmax))
        centers.append((a + b) // 2)
    return np.array(centers), np.array(info_vals)

def write_average_HFD_values_for_each_age_range(sex, num_k, kmax, cycle_step, higuchi_average_per_each_age_group):
    with open('output/{0}_HFD_average_of_ECG_per_age_range_kmax_{1}_cycle_step_{2}_num_k_{3}_full.csv'.format(sex, kmax, cycle_step, num_k), 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for age_group in higuchi_average_per_each_age_group.keys():
            spamwriter.writerow([m2.age_groups[age_group], f"{higuchi_average_per_each_age_group[age_group]:.3f}".replace('.', ',')])

def write_HFD_calculated_value_to_csv(sex, id, info, kmax, step_cycle, knum):
    # ECG 1 and 2 simulationusly

    file_path = 'output/{0}_HFD_all_ECG_calculated_kmax_is_{1}_step_cycle_{2}_num_k_{3}.csv'.format(sex, kmax, step_cycle, knum)

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
            for i in range(1, windows_count + 1, 1):
                list += ['k{0}'.format(i), 'b{0}'.format(i), 'D{0}'.format(i), 'p-value linear {0}'.format(i),
                         'R_score{0}'.format(i), 'AIC_linear{0}'.format(i),
                         'kef x^2 ({0})'.format(i), 'kef x ({0})'.format(i),'kef 1 ({0})'.format(i),
                         'p-value quadr {0}'.format(i), 'R_score quadr {0}'.format(i), 'AIC_quadr{0}'.format(i)]
            spamwriter.writerow(list)


        # Добавляем новые строки


        list = [id]
        for i in range(0, windows_count, 1):
            # info[i][0] - i-th window k parameter
            # info[i][1] - i-th window b parameter
            # info[i][2] - i-th window D parameter
            # info[i][3] - i-th window p-value linear parameter
            # info[i][4] - i-th window R_square parameter
            # info[i][5] - i-th window AIC parameter
            # info[i][6] - i-th window ax^2 parameter
            # info[i][7] - i-th window by parameter
            # info[i][8] - i-th window c parameter
            # info[i][9] - i-th window p-value squared parameter
            # info[i][10] - i-th window R_square quad
            # info[i][11] - i-th window AIC quad


            list += [f"{info[i][0]:.3f}", f"{info[i][1]:.3f}", f"{info[i][2]:.3f}", f"{info[i][3]:.3f}",
                     f"{info[i][4]:.3f}", f"{info[i][5]:.3f}", f"{info[i][6]:.3f}", f"{info[i][7]:.3f}",
                     f"{info[i][8]:.3f}", f"{info[i][9]:.3f}", f"{info[i][10]:.3f}", f"{info[i][11]:.3f}"]
        spamwriter.writerow(list)





        """, RECORD.DATABASE[type_of_ecg_cut][key].Sex,
                                 RECORD.DATABASE[type_of_ecg_cut][key].BMI]"""

def extract_id_from_filename(filename):
    # Используем регулярное выражение для извлечения числового индекса
    match = re.search(r'_(\d+)\.txt', filename)
    if match:
        index = match.group(1)
        return index

def find_maximum_id_in_full_ECG_id_to_info_file(kmax, step_cycle):
    """If you open csv file with full ECG id to info parameters, it finds id of row with maximum id

        output: id
    """

    file_path = 'output/{0}_HFD_all_ECG_calculated_kmax_is_{1}_step_cycle_{2}.csv'.format("both_sexes", kmax, step_cycle)

    from pathlib import Path

    my_file = Path(file_path)
    if not my_file.is_file():
        return -1

    max_id = -1
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # пропускаем заголовок, если он есть
        for row in reader:
            try:
                value = int(row[0])
                if value > max_id:
                    max_id = value
            except ValueError:
                continue

    print("Максимальный id:", max_id)
    #max_id.lstrip("0")  # '1034' (без нулей впереди)
    return max_id  # '1034' (без нулей впереди)

def create_full_ECG_id_to_info_file(kmax, step_cycle,num_k):

    try:
        cleaned_signals_filenames = list_files_with_cleaned_signal()    # Files of cleaned and r_peaks must be count equally
        r_peaks_filenames = list_files_with_r_peaks()                   # and have corresponding id's.

        ids_csf = [extract_id_from_filename(filename) for filename in cleaned_signals_filenames]  # Extract corresponding
                                                                                              # id's for
                                                                                              # cleaned_signals_filenames
                                                                                              # above.
        ids_rp = [extract_id_from_filename(filename) for filename in r_peaks_filenames] # Extract corresponding
                                                                                    # id's for
                                                                                    # r_peaks_filenames
                                                                                    # above.
        if ids_csf != ids_rp:
            raise FileNotFoundError("Error! Filenames of cleaned signals and r-peaks must be corresponding!")
    except NameError as e:
        print("NameError:", e)
    ids = ids_csf

    #tep_cycle = 50
    #kmax_list = [10000, 16000, 25000]
    #kmax = kmax_list[2]
    min_id = find_maximum_id_in_full_ECG_id_to_info_file(kmax, step_cycle) + 1


    # HFD = {} # Dictionary with HFD for each id

    for i in range(min_id, len(ids), 1):
        id = ids[i]
        _, cleaned_time_series = extract_from_file_cleaned_signal_time_series(cleaned_signals_filenames[i])
        peaks = extract_from_file_r_peaks_time_series(id, r_peaks_filenames[i])

        normalized_cleaned = zscore_normalize(cleaned_time_series)

        #  Видалити значення менші нуля на початку peaks
        while peaks and peaks[0] < 0:
            peaks.pop(0)

        min_length = 480501                 #Min length of ECG
        cycle_duration = 1500               #Max duration of heart cycle
        cycles_in_window = 100


        min_cycles = min_length / cycle_duration
        get_cycles = int(min_cycles / cycles_in_window) * cycles_in_window
        selected_R_peak_cycles = peaks[:get_cycles] #Беремо задану кількість серцевих циклів
        print(cleaned_signals_filenames[i])
        print(selected_R_peak_cycles)
        print("Cycles: " + str(get_cycles))

        print("Length: " + str(len(normalized_cleaned)))
        print("Peaks: " + str(len(selected_R_peak_cycles)))
        # Тут закінчив.

        #Firstly kmax=1500, step_cycles = 20
        centers_idx, info = windowed_hfd_cycles(normalized_cleaned, selected_R_peak_cycles, id, num_k=num_k, n_cycles=100, step_cycles=step_cycle,
                                               kmax=kmax)

        #info - list with lists with [k, b, D, R_score]
        #hfd_mean = np.mean(hfd)
        #print(hfd_mean)

        #write_HFD_calculated_value_to_csv("both sexes", id, hfd_mean)
        write_HFD_calculated_value_to_csv("both_sexes", id, info, kmax, step_cycle, num_k)
        # HFD[id] = hfd_mean
        # Для времени: t_centers = centers_idx / fs

        if peaks == None:
            break
"""ВНИМАНИЕ! R-пики могут быть отрицательны"""

def load_id_to_hfd(kmax, step_cycle, num_k):
    file_path = 'output/{0}_HFD_all_ECG_calculated_kmax_is_{1}_step_cycle_{2}_num_k_{3}_full.csv'.format("both_sexes", kmax,
                                                                                          step_cycle, num_k)



    import pandas as pd

    # читаем CSV
    df = pd.read_csv(file_path,sep=";")

    # фиксированные колонки (которые не группируются)
    fixed_cols = ["id"]

    # все остальные (которые идут пятёрками)
    other_cols = [c for c in df.columns if c not in fixed_cols]

    # разбиваем на группы по 5
    groups = [other_cols[i:i + 12] for i in range(0, len(other_cols), 12)]

    id_to_hfd = {}

    higuches = []
    # пример обхода по группам
    for idx, group in enumerate(groups, start=1):
        print(f"\n=== Group {idx} ===")
        print("Columns:", group)

        # достать данные конкретной группы
        #sub_df = df[fixed_cols + group]
        higuches.append(df[group[2]])

        # info[i][0] - i-th window k parameter
        # info[i][1] - i-th window b parameter
        # info[i][2] - i-th window D parameter
        # info[i][3] - i-th window R_square parameter
        # info[i][3] - i-th window p-value parameter

    for i in range(len(higuches[0])):
        higuches_line = []
        for j in range(len(higuches)):
            higuches_line.append(higuches[j][i])


        averaged_hfd = np.mean(higuches_line)

        id_to_hfd[f"{df['id'][i]:04d}"] = averaged_hfd

    #print(id_to_hfd)
        # можно обрабатывать дальше — например, сохранить отдельно
        # sub_df.to_csv(f"group_{idx}.csv", index=False)

    return id_to_hfd


    """
    with open('output/both_sexes_HFD_all_ECG_calculated.csv', newline='') as csvfile:
        spamreader = csv.DictReader(csvfile, delimiter=';', quotechar='|')
        for row in spamreader:
            id_ = f"{int(row['id']):04d}"
            hfd_value = float(row['HFD'])  # если числа с запятой
            id_to_hfd[id_] = hfd_value
    return 
    """

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

    step_cycle = 50
    kmax_list = [10000, 16000, 25000]
    kmax = kmax_list[2]
    num_k=50

    #create_full_ECG_id_to_info_file(kmax, step_cycle, num_k)


    id_to_hfd = load_id_to_hfd(kmax, step_cycle, num_k)

    print(id_to_hfd)

    keys = id_to_hfd.keys()
    male_ids, female_ids = m2.classify_ids_by_sex(id_to_hfd.keys())

    print(male_ids)
    print(female_ids)

    male_id_ageRangeIndex_dict, female_id_ageRangeIndex_dict = m2.get_age_ranges_for_male_and_female(keys, male_ids, female_ids)

    male_age_range_to_mean_hfd = age_range_agregation(id_to_hfd, male_id_ageRangeIndex_dict)
    female_age_range_to_mean_hfd = age_range_agregation(id_to_hfd, female_id_ageRangeIndex_dict)

    #male_age_range_to_count = age_range_agregation_count(id_to_hfd, male_id_ageRangeIndex_dict)
    #female_age_range_to_count = age_range_agregation_count(id_to_hfd, female_id_ageRangeIndex_dict)
    #m2.write_number_of_ECGs_per_age_range_for_both_HFD("male", male_age_range_to_count)
    #m2.write_number_of_ECGs_per_age_range_for_both_HFD("female", female_age_range_to_count)

    write_average_HFD_values_for_each_age_range("male",num_k, kmax, step_cycle, male_age_range_to_mean_hfd)
    write_average_HFD_values_for_each_age_range("female",num_k, kmax, step_cycle, female_age_range_to_mean_hfd)
    #write_average_HFD_values_for_each_age_range("both_sexes", age_to_mean_hfd)




