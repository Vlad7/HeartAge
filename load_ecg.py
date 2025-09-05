import pandas as pd
import matplotlib as plt

# Import data
path = r'D:\SCIENCE\Datasets\autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0\0001.hea'
df = pd.read_csv(path)
print(df)

# plot
plt.title("ECG signal - 1000 Hz")
plt.plot(df.ecg[0:5000])
plt.xlabel('Time (milliseconds)')
plt.show()