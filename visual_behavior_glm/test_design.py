from GLM_fit_tools import toeplitz
import matplotlib.pyplot as plt
import numpy as np

events = np.zeros((10,))
events[2] = 1
events[3] = 1
events[4] = 1

kls = 5
kernel = toeplitz(events, kls, 0).T
print(kernel)

plt.clf()
plt.imshow(kernel, aspect='auto', interpolation='nearest')
plt.xlabel('Position of window (reversed)')
plt.ylabel('Time bin of output')
plt.colorbar()
plt.savefig('/home/saaketh.medepalli/visual_behavior_glm/visual_behavior_glm/omission_design3.png') 
