from specxplore.specxplore_data import SpectraDF
import numpy as np
import pandas as pd
from collections import namedtuple

mz = np.array([100, 130, 145], dtype = np.double)
intensity = np.array([0.6, 0.01, 1], dtype = np.double)
identifier = np.array([0,1,2], dtype = np.int64)
df = pd.DataFrame({"intensity" : intensity, "mass-to-charge-ratio" : mz, "identifier" : identifier})


print(dict(zip(("identifier", "mass-to-charge-ratio", "intensity"), [np.int64, np.double, np.double])))



#expected_column_types_class = namedtuple("expected_column_types", ['id', 'mz', 'int'])
#expected_column_types = expected_column_types_class(id=np.int64, mz=np.double, int = np.double)
#print(expected_column_types)
print("IGNORING THE _data INTERNAL VARIABLE INDICATOR CAN LEAD TO NONSENSE SIDE EFFECTS IN THE OBJECT.")
print("HOWEVER, THE GETTER APPROACH IS THE EASIEST WAY TO GET IMMUTABLE FEEL & ALERT PEOPLE TO NOT MODIFY PAST INIT.")
print(SpectraDF(df))
newdata = SpectraDF(df)
newdata._data["identifier"] = ["a", "bunch of", "nonsense"] # <-- THIS OPERATION WORKS, BUT IT IS CONSIDERED BAD PRACTICE TO MODIFY INTERNAL CLASS OBJECTS!
print(newdata._data)
print(newdata.get_column_as_np("identifier"))
print(newdata.get_data())

print("USING GETTERS AVOIDS MUTABILITY BOTTLENECKS AND OVERWRITING PANDAS DF COLUMNS")
df = pd.DataFrame({"intensity" : intensity, "mass-to-charge-ratio" : mz, "identifier" : identifier})
newdata = SpectraDF(df)
data = newdata.get_data()
data["identifier"] = ["a", "bunch of", "nonsense"]
print(data)
print(newdata)
# This line breaks... but the previous string replacement does not lead to any errors.
#newdata.data = df