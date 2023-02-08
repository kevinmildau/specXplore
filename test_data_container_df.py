from specxplore.specxplore_data import MultiSpectrumDataFrameContainer
import numpy as np
import pandas as pd
from collections import namedtuple

mz = np.array([100, 130, 145], dtype = np.double)
intensity = np.array([0.6, 0.01, 1], dtype = np.double)
identifier = np.array([0,1,2], dtype = np.int64)
df = pd.DataFrame({"intensity" : intensity, "mass-to-charge-ratio" : mz, "identifier" : identifier})


print(dict(zip(("identifier", "mass-to-charge-ratio", "intensity"), [np.int64, np.double, np.double])))



expected_column_types_class = namedtuple("expected_column_types", ['id', 'mz', 'int'])
expected_column_types = expected_column_types_class(id=np.int64, mz=np.double, int = np.double)
print(expected_column_types)

print(MultiSpectrumDataFrameContainer(df))
newdata = MultiSpectrumDataFrameContainer(df)
newdata.data["identifier"] = ["1", "2", "3"]
print(newdata.data)
print(newdata.get_column_as_np("identifier"))
print(newdata.get_data())

# This line breaks... but the previous string replacement does not lead to any errors.
#newdata.data = df