import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

Wave = np.loadtxt(
    "./EMwavedata10000/traces10000.csv",
    dtype="float",
    delimiter=",",
)
Ave_Wave = np.mean(Wave, axis=0)
kernel = np.ones(100) / 100
smoothed = np.convolve(Ave_Wave, kernel, mode="valid")
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(np.mean(Wave, axis=0))
ax1.set_xlim(1000, 3500)
ax1.set_title("Mean Wave")
fig1.savefig("running-mean.jpg")
