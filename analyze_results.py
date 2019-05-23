import pandas as pd

results = pd.concat([pd.read_csv('testoutput.csv/chain-0.csv').iloc[1000:, :],
                     pd.read_csv('testoutput.csv/chain-1.csv').iloc[1000:, :]])

significant_pe = []
significant_low = []
significant_high = []
significant_name = []

for item in list(results):
    if results[item].quantile(.025) > 0 or results[item].quantile(.975) < 0:
        significant_name.append(item)
        significant_low.append(results[item].quantile(.025))
        significant_high.append(results[item].quantile(.975))
        significant_pe.append(results[item].mean())

for i in range(len(significant_name)):
    print(significant_name[i] + ': ' + str(significant_low[i]) + ' ' + str(significant_pe[i]) + ' ' + str(significant_high[i]))