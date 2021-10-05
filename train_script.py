import previsionio as pio
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import os

pio.client.init_client(
    token=os.environ['PIO_MASTER_TOKEN'],
    prevision_url='https://int.prevision.io')

with open('dset_config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

train = pio.Dataset(config['train'], name='churn_train')
test = pio.Dataset(config['test'], name='churn_test')

train_data = train.data
test_data = test.data

TARGET = 'contract_ended'

clr = make_pipeline(OrdinalEncoder(), LogisticRegression())
clr.fit(train_data.drop(TARGET, axis=1), train_data[TARGET])

initial_type = [('float_input', FloatTensorType([None, train_data.drop(TARGET, axis=1).shape[1]]))]
onx = convert_sklearn(clr, initial_types=initial_type)
with open('logreg_churn.onnx', 'wb') as f:
    f.write(onx.SerializeToString())

yaml_dict = {
    'class_names': train_data[TARGET].unique().tolist(),
    'input': list(train_data.drop(TARGET, axis=1).keys())
}

with open('logreg_churn.yaml', 'w') as f:
    yaml.dump(yaml_dict, f)

p = pio.Project(_id='60d36081b4efdf001c68feb5', name='playground')

exp = p.create_external_classification(experiment_name='churn',
                                       dataset=train,
                                       holdout_dataset=test,
                                       target_column=TARGET,
                                       external_models=[
                                           ('logreg', 'logreg_churn.onnx', 'logreg_churn.yaml')
                                       ])
