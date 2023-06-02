import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,20,10000)
# y = np.arctan((c+10)*(x))/(0.01+(x)) *(2/np.pi) * c
c_b = 0.001
c_r = 6
y = np.arctan((x**2))/(c_b + x) * c_r

f = 1/(x+1) * c_r

z = 1/x * c_r * (np.pi/2)

#

plt.figure()
plt.plot(x,z,label="1/x")
plt.plot(x,y,label="tan_func")
plt.plot(x,f,label="f")
plt.ylim((0,np.max(y)))
plt.title("angular rate")
plt.legend()
plt.show()

plt.figure()
plt.plot(x,z*x,label="1/x")
plt.plot(x,y*x,label="tan_func")
plt.plot(x,f*x,label="f")
plt.ylim((0,np.max(y*x)+2))
plt.title("centripetal acceleration")
plt.legend()
plt.show()

# plt.figure()
# plt.plot(x,f,label="f")
# plt.title("centripetal acceleration")
# plt.show()
