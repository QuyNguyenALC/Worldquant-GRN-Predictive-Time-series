import os
from joblib import load
import numpy as np

# #############################################################################
# Load model
#print("Loading model from working directory")
clf = load('PredictRevenue.joblib') 
# #############################################################################
# Run inference
forecast = clf.forecast(steps=14)[0]
result = [] 
for i in forecast:
    print(np.exp(i))
    result.append(np.exp(i)*7)
print("Predict revenue...")
print(result)

