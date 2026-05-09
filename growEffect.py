import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg"

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

fig, ax = plt.subplots()
x = np.linspace(0, 2*np.pi, 100)
line, = ax.plot(x, np.sin(x))

def grow(frame):
    line.set_ydata(np.sin(x) * frame/50)  # gradually grows amplitude
    return line,

ani = animation.FuncAnimation(fig, grow, frames=50, interval=100, blit=True)
plt.show()