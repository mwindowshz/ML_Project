import pandas as pd
import numpy as np
import matplotlib as plt

ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))

plt.interactive(False)
ts = ts.cumsum()

ts.plot()
plt.pyplot.show()