import pandas as pd
import numpy as np

# Пример данных
s = pd.Series([np.nan, np.nan, 10, 20, np.nan, 30, np.nan, np.nan])

# Убираем только NaN в начале и в конце
s_trimmed = s[s.first_valid_index():s.last_valid_index() + 1]

print(s_trimmed)
