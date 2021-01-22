import matplotlib.pyplot as plt
import os

keys = ["1%", "2%", "4%", "8%", "16%", "32%", "64%"]
values = [6.44, 59.72, 94.77, 98.37, 97.8, 99.59, 99.73]
plt.figure()
plt.bar(keys, values, color='b')
plt.xlabel('Percent of commands used for training')
plt.ylabel('Accuracy on new commands(%)')
file_path = os.path.join('analysis_scripts','exp1.png')
plt.savefig(file_path)