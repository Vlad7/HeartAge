# This is an ECG Higuchi script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Age groups

age_groups = {'1': '18 - 19',
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



from IPython.display import display
import numpy as np
import wfdb
import HiguchiFractalDimension.hfd
import csv

#######################################################################################################################

# Path to dataset of ECG
path = 'D:/SCIENCE/Datasets/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0'

csv_info_file = 'subject-info.csv'

#######################################################################################################################

def open_info_file():

    """ Open csv info file, print header and information for each record """

    with open(path+'/'+ csv_info_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                print(f'\tId: {row[0]}; Age_group: {age_groups[row[1]]}; Sex: {row[2]}; BMI: {row[3]}; Length: {row[4]}; Device: {row[5]}.')
                open_record(row[0])
                line_count += 1
        print(f'Processed {line_count} lines.')

def open_record(id):

    """ Open each record by Id """

    # wfdb.rdrecord(... [0, 1] - first two channels (ECG 1, ECG 2); [0] - only first ECG
    # 0 - The starting sample number to read for all channels (point from what graphic starts).
    # None - The sample number at which to stop reading for all channels. Reads the entire duration by default.

    record = wfdb.rdrecord(
        path + '/' + id, 0, 500000, [0, 1])

    sequence = []

    # print(record.p_signal)

    for x in record.p_signal:

        # Use only first ECG
        sequence.append(x[0])

    #print(sequence)

    HFD = HiguchiFractalDimension.hfd(np.array(sequence), opt=False)
    print("Higuchi fractal dimension: " + str(HFD))

    #print(record)
    wfdb.plot_wfdb(record, title='Record' + id + ' from Physionet Autonomic ECG')
    #display(record.__dict__)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #myarray = np.fromfile("D:/Projects/ECGHiguchi/mit-bih-arrhythmia-database-1.0.0/101.dat", dtype=float)

    #for i in range (0, len(myarray)):
    #    print(myarray[i])

    #record = wfdb.rdrecord('D:/Projects/ECGHiguchi/mit-bih-arrhythmia-database-1.0.0/102', 4)
    #wfdb.plot_wfdb(record, title='Record a01 from Physionet Apnea ECG')
    #display(record.__dict__)



    x = np.random.randn(10000)
    #y = np.empty(9900)
    #for i in range(x.size - 100):
    #    y[i] = np.sum(x[:(i + 100)])

    ## Note x is a Guassian noise, y is the original Brownian data used in Higuchi, Physica D, 1988.

    #print(HiguchiFractalDimension.hfd(x, opt=False))
    # ~ 2.00
    #hfd.hfd(y)  # ~ 1.50


    print_hi('PyCharm')

    open_info_file()







# See PyCharm help at https://www.jetbrains.com/help/pycharm/
