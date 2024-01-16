# generate loss curve from Faster R-CNN training

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#path = "output/metrics.json"
path = "MMBCReco/src/output/metrics.json"

jFile = pd.read_json(path, lines=True)
data = jFile["total_loss"]
xaxis = np.linspace(0, len(data), len(data))

plt.plot(xaxis, data)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.savefig("MMBCReco/src/results/lossCurve.png")
plt.xlabel("iterations")
plt.ylabel("loss")
plt.show()