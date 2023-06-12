import numpy as np
import matplotlib.pyplot as plt

L = 5
beta = np.linspace(-np.pi, np.pi,1000)
lr = 1.2
delta = np.arctan2(L*np.tan(beta), lr)

plt.figure()
plt.plot(beta,delta,label="")
# plt.ylim((0,np.max(y)))
# plt.title("angular rate")
plt.legend()
plt.show()