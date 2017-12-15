import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

cumm_awards = pd.read_csv("total_rewards.csv")

data=np.array(cumm_awards)
plt.plot(range(1,data.shape[0]+1),data,label="Best Model")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title('Total Rewards per Game')
plt.legend()
plt.savefig("total_reward_with_features.png")
plt.show()
plt.gcf().clear()



last_awards = pd.read_csv("last_rewards.csv")
data=np.array(last_awards)
plt.plot(range(1,data.shape[0]+1),data,label="Best Model")
plt.plot(range(1,data.shape[0]+1),6*np.ones(data.shape[0]),label="Baseline")
plt.legend()
plt.xlabel("Episodes")
plt.ylabel("Final Reward")
plt.title('Final Rewards per Game')
plt.savefig("final_reward_with_features.png")
plt.show()
