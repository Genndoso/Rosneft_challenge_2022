import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import numpy as np



import warnings
warnings.filterwarnings("ignore")


#data_path = './Входные данные/signals.csv'
data_path = 'signals.csv'

# ./Входные данные/signals.csv', header=None

def solution(data_path):
    df = pd.read_csv(data_path, header=None)
    labeled = df[df[5003] != -1]
    coef_per_cluster = []
    for i in range(0, 9):
        data = labeled[labeled.iloc[:, 5003] == i]

        coef = coef_finding(data,value=False)
        coef_per_cluster.append(coef)
     #   print(f'Coefficients of {i} group of clusters {coef}')

### finding similarity between values and clusterf

     #df[df[5003] == -1]

    for i in range(0, len(df[df[5003] == -1])):
        try:
            df[df[5003] == -1].iloc[i, 5003] = similarity_search(df[df[5003] == -1].iloc[i, 3:5002], coef_per_cluster, value=True)
        except (ValueError, np.linalg.LinAlgError):
            continue

    return df




def similarity_search(data,coef_per_cluster,value=True):
    norm = coef_finding(data,value)
    for i,coef in enumerate(coef_per_cluster):
     #   print(i)
        if i == 0:
            cluster = 0
            coef
            minus_norm = np.linalg.norm(norm-coef)
        #    print(minus_norm)
        else:
            sec_norm = np.linalg.norm(norm-coef)
         #   print(sec_norm)
            if sec_norm<minus_norm:
                minus_norm = sec_norm
                cluster = i
    return cluster




def coef_finding(data, value=True):
    coef = []
    if value:
        model = sm.tsa.ARMA(data, order=(10, 1, 0))
        model_fit = model.fit(transparams=False)
        coef.append(model_fit.arparams)
        return coef
    else:
        for i in range(0, len(data)):
            # print(i)
            try:
                model = sm.tsa.ARMA(data.iloc[i, 3:5003], order=(10, 1, 0))
                model_fit = model.fit(transparams=False)
            except ValueError:
                model = sm.tsa.ARMA(data.iloc[i, 3:5003], order=(5, 1))
                model_fit = model.fit(transparams=False)
            coef.append(model_fit.arparams)
        mean = np.zeros_like(coef[0])
        for i in range(0, len(coef)):
            a = coef[i]
            for j in range(0, len(a)):
                mean[j] += a[j]
        return mean / len(data)


if __name__ == '__main__':
    solution(data_path)
