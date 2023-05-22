import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,20,10000)
c = 10
# y = np.arctan((c+10)*(x))/(0.01+(x)) *(2/np.pi) * c
y = np.arctan((x**2))/(0.001 + x) *c

f = 1/(x+1) * c

z = 1/x * c * (np.pi/2)

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
