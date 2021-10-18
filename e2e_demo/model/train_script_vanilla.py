from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

train = pd.read_csv('gs://prevision-public/churn_train_vivatech.csv')
test = pd.read_csv('gs://prevision-public/churn_test_vivatech.csv')

TARGET = 'contract_ended'

clr = make_pipeline(OrdinalEncoder(), LogisticRegression())
clr.fit(train.drop(TARGET, axis=1), train[TARGET])

initial_type = [('float_input', FloatTensorType([None, train.drop(TARGET, axis=1).shape[1]]))]
onx = convert_sklearn(clr, initial_types=initial_type)
with open('artifacts/logreg_churn.onnx', 'wb') as f:
    f.write(onx.SerializeToString())
