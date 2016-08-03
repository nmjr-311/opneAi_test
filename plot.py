import matplotlib.pyplot as plt
import pandas as pd

name = 'success_trial'
df = pd.read_csv('logs/yama/log_success.csv')

plt.plot(df['gen'], df['best_reward'])
plt.grid()

plt.xlabel('generation')
plt.ylabel('best reward')

#plt.xlim(0, 2000)
plt.ylim(0, 1000)

plt.savefig(name+'.pdf')
plt.show()
plt.close()
