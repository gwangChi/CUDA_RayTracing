import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.image as mpimg

# Make data.
X = np.arange(-0.5, 0.5, 0.0025)
Y = np.arange(-0.5, 0.5, 0.0025)
X, Y = np.meshgrid(X, Y)
filenames = []

Z = np.loadtxt("1000000000.dat")
#Z = np.loadtxt("1000.dat")
plt.imshow(Z, cmap='gray', origin ='lower')
plt.title("1 Billion Rays Serial Implementation")
plt.show()