import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
data = pd.read_csv('train_reward/original/original_reward_per_episode.csv')

# 绘制图像
plt.figure(figsize=(10, 6))
plt.plot(data['episode'], data['reward'], marker='o', label='Reward')

# 设置标题和轴标签
plt.title('Episode vs Reward', fontsize=16)
plt.xlabel('Episode', fontsize=14)
plt.ylabel('Reward', fontsize=14)

# 添加网格和图例
plt.grid(alpha=0.5)
plt.legend(fontsize=12)

# 显示图像
plt.tight_layout()
plt.show()
# train_reward/original/original_reward_per_episode.csv