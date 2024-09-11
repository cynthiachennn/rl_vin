import numpy as np
import matplotlib.pyplot as plt

train_file = 'loss/2024-09-10-23-55-53_train_loss.npy'
val_file = 'loss/2024-09-10-23-55-53_val_loss.npy'
train_loss = np.load(train_file)
val_loss = np.load(val_file)

print(len(train_loss), len(val_loss))
plt.title(train_file)
# plt.plot(range(0, len(train_loss) * 1000, 1000), train_loss, label='train')
# plt.plot(range(0, len(train_loss) * 1000 -5000, int(len(train_loss)/len(val_loss) * 1000)), val_loss, label='val')
# plt.plot(actual_train_loss, label='train')
plt.plot(train_loss, label='train')
plt.plot(val_loss, label='val')

print(np.argmin(train_loss))
print(np.argmin(val_loss))
plt.legend()
plt.show()