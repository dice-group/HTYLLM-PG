import numpy as np
import sys

arr = np.load(file=sys.argv[1])

print(arr[:20])