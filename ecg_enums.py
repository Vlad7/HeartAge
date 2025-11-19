import enum
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

"""
class TypeOfECGCut(enum.Enum):

    full = 1
    start = 2
    middle = 3
    end = 4
"""

class TypeOfTimeSeries(enum.Enum):

    full_ecg = 1
    hrv_ecg = 2
    hrv_AIC_ecg = 3

class Gender(enum.Enum):
    male = 1
    female = 2
    both_sexes = 3

class Dataset(enum.Enum):
    germany = 1
    ukraine = 2