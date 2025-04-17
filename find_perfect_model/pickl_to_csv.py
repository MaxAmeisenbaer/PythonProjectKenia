import pandas as pd
import os.path
import openpyxl

number_of_files = len([name for name in os.listdir('.') if os.path.isfile(name)])
print(number_of_files)
for count in []:
    filename = f"SHa-nit-prediction-model_{count}.pkl"
    df = pd.read_pickle(filename)
    df.to_excel(f"SHa-nit-prediction-model_{count}.xlsx")

