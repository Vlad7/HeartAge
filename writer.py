import ecg_enums as ee
import os
import csv

def write_average_HFD_values_for_each_age_range(sex, num_k, kmax, window_step, higuchi_average_per_each_age_group, method, time_series_type):


    file_path = None

    if time_series_type == TypeOfTimeSeries.full_ecg.name:
        file_path = 'output/{0}_HFD_all_ECG_calculated_kmax_is_{1}_step_cycle_{2}_num_k_{3}.csv'.format(sex, kmax,
                                                                                                        window_step,
                                                                                                        num_k)
    elif time_series_type == 'hrv_ecg':
        file_path = 'output/hrv/{0}_HFD_HRV_ECG_calculated_num_k_is_{1}_kmax_is_{2}_window_step_is_{3}.csv'.format(sex,
                                                                                                                   num_k,
                                                                                                                   kmax,
                                                                                                                   window_step)
    elif time_series_type == 'hrv_AIC_ecg':
        file_path = 'output/hrv/{0}_HFD_HRV_AIC_ECG_calculated_num_k_is_{1}_kmax_is_{2}_window_step_is_{3}.csv'.format(sex,
                                                                                                                   num_k,
                                                                                                                   kmax,
                                                                                                                   window_step)

    with open(file_path, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for age_group in higuchi_average_per_each_age_group.keys():
            spamwriter.writerow([m2.age_groups[age_group], f"{higuchi_average_per_each_age_group[age_group]:.3f}".replace('.', ',')])

            hpar.write_HFD_calculated_info_to_csv(ee.Gender.both_sexes.name, ee.TypeOfTimeSeries.hrv_ecg, key, info, k_max, None, num_k_value, 'gerontology')


  w.write_HFD_calculated_info_to_csv(ee.Gender.both_sexes, ee.TypeOfTimeSeries.hrv_ecg, key, info, k_max,
                                                  None, num_k_value, dataset)

def write_HFD_calculated_info_to_csv(sex, time_series_type, id, info, kmax, window_step, num_k, source):

    file_path = None

    if time_series_type == ee.TypeOfTimeSeries.full_ecg:
        if source == ee.Dataset.ukraine:
            file_path = 'output/{0}/{1}_HFD_all_ECG_calculated_kmax_is_{2}_step_cycle_{3}_num_k_{4}.csv'.format(
                source.name, sex.name, kmax, step_cycle, knum)
        else:
            file_path = 'output/{0}/{1}_HFD_all_ECG_calculated_kmax_is_{2}_step_cycle_{3}_num_k_{4}.csv'.format(
                source.name, sex.name, kmax, step_cycle, knum)

        file_path = 'output/{0}_HFD_all_ECG_calculated_kmax_is_{1}_step_cycle_{2}_num_k_{3}_{4}.csv'.format(sex.name,
                                                            kmax, step_cycle, knum, source.name)
    elif time_series_type == ee.TypeOfTimeSeries.hrv_ecg:
        if source == ee.Dataset.ukraine:
            file_path = 'output/hrv/{0}/{1}_HFD_HRV_ECG_calculated_num_k_is_{2}_kmax_is_{3}_window_step_is_{4}.csv'.format(source.name, sex.name,
                                                            num_k, kmax, window_step)
        else:
            file_path = 'output/hrv/{0}/{1}_HFD_HRV_ECG_calculated_num_k_is_{2}_kmax_is_{3}_window_step_is_{4}.csv'.format(
                source.name, sex.name,
                num_k, kmax, window_step)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, file_path)


    # Check, if file exist or empty
    file_exists = os.path.isfile(file_path)
    file_empty = not file_exists or os.path.getsize(file_path) == 0

    with open(file_path, 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

        windows_count = len(info)

        # If file is empty, write title
        if file_empty:

            list = ['id']
            for i in range(1, windows_count + 1, 1):
                list += ['k{0}'.format(i), 'b{0}'.format(i), 'D{0}'.format(i), 'p-value linear {0}'.format(i),
                         'R_score{0}'.format(i), 'AIC_linear{0}'.format(i),
                         'kef x^2 ({0})'.format(i), 'kef x ({0})'.format(i),'kef 1 ({0})'.format(i),
                         'p-value quadr {0}'.format(i), 'R_score quadr {0}'.format(i), 'AIC_quadr{0}'.format(i)]
            spamwriter.writerow(list)


        # Add new rows

        list = [id]

        for i in range(0, windows_count, 1):
            # info[i]['k'] - i-th window k parameter
            # info[i]['b'] - i-th window b parameter
            # info[i]['D'] - i-th window D parameter
            # info[i]['p-value linear'] - i-th window p-value linear parameter
            # info[i]['R_score'] - i-th window R_square parameter
            # info[i]['AIC_linear'] - i-th window AIC parameter
            # info[i]['kef x^2'] - i-th window ax^2 parameter
            # info[i]['kef x'] - i-th window by parameter
            # info[i]['kef 1'] - i-th window c parameter
            # info[i]['p-value quadr'] - i-th window p-value squared parameter
            # info[i]['R_score quadr'] - i-th window R_square quad
            # info[i]['AIC_quadr'] - i-th window AIC quad


            list += [f"{info[i]['k']:.3f}", f"{info[i]['b']:.3f}", f"{info[i]['D']:.3f}", f"{info[i]['p-value linear']:.25f}",
                     f"{info[i]['R_score']:.3f}", f"{info[i]['AIC_linear']:.3f}", f"{info[i]['kef x^2']:.3f}", f"{info[i]['kef x']:.3f}",
                     f"{info[i]['kef 1']:.3f}", f"{info[i]['p-value quadr']:.25f}", f"{info[i]['R_score quadr']:.3f}", f"{info[i]['AIC_quadr']:.3f}"]

        spamwriter.writerow(list)





        """, RECORD.DATABASE[type_of_ecg_cut][key].Sex,
                                 RECORD.DATABASE[type_of_ecg_cut][key].BMI]"""
