import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(filepath_or_buffer='./menu.csv')

max_sodium = df['Sodium'].max()
sodium_max_df = df[df['Sodium'] == max_sodium]

print(sodium_max_df.head())


#sns.swarmplot(x='Category', y='Sodium', data=df)
sns.scatterplot(x='Category', y='Sodium', data=df)

plt.show()

print("")
