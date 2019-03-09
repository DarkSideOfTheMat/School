import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt 
import matplotlib.style as style
from matplotlib import rc
import seaborn as sns
from math import sqrt
import statsmodels.formula.api as smf

style.use('seaborn-darkgrid')

rc('text', usetex=True)

EWJ_data = Path.cwd() / 'data' / 'EWJ.csv'

df = pd.read_csv(EWJ_data, parse_dates = [1], infer_datetime_format = True)
df['Date'] = pd.to_datetime(df['Date'])

day = 1
week = 5
month = 21
quarter = 63
half = 126
year = 252
intervals = [day, week, month, quarter, half, year]

def returns_calculator(df , delta = 1):
	result = []
	i = delta
	while i in df.index:
		curr_close, prev_close = df['Close'][i], df['Close'][i - delta]
		return_rate = 100 * (curr_close - prev_close)/prev_close
		result.append(return_rate)
		i += delta
	return result

results = pd.DataFrame()

results['tau'] = np.array([i/252 for i in intervals])
results['Sqrttau'] = results['tau'].apply(sqrt)
results['StdDev'] = np.array([np.std(returns_calculator(df, i)) for i in intervals])

mod = smf.ols(formula = 'StdDev ~ 1 + Sqrttau', data = results)
res = mod.fit()
print(res.summary())

fig, ax = plt.subplots()

sns.regplot(
	x = 'Sqrttau', y = 'StdDev', line_kws = {'linestyle': '--', 'color': 'blue'},
	 scatter_kws = {'color': 'black'}, data = results, ax = ax
	)
ax.set_title('Diffusion of Japanese Returns (Square Root Scale)')
ax.set_xlabel(r'$\sqrt{Time(Year)}$')
ax.set_ylabel('Standard Deviation of Returns (\%)')
plt.show()

