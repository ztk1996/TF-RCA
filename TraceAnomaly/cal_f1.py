from unittest import result
import pandas as pd


df = pd.read_csv('result_real_data/p_r_curve.csv')
prec = df['prec']
recall = df['recall']

df['f1'] = (2 * prec * recall) / (prec + recall)

df.sort_values(by=['f1'],ascending=False).to_csv('result_f1.csv')