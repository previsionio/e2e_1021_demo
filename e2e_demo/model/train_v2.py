from . import make_yaml_dict
import previsionio as pio
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import os
import datetime

pio.client.init_client(
    token=os.environ['PIO_MASTER_TOKEN'],
    prevision_url='https://pistre.prevision.io')

with open('e2e_demo/model/artifacts/dset_config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

train_dset = pio.Dataset(config['train'], name='churn_train')
test_dset = pio.Dataset(config['test'], name='churn_test')

train = train_dset.data
test = test_dset.data

TARGET = 'contract_ended'

models = [DecisionTreeClassifier(), LogisticRegression(),
          RandomForestClassifier(), ExtraTreesClassifier()]

pio_model_list = []

for model in models:
    clr = make_pipeline(OrdinalEncoder(), model)
    clr.fit(train.drop(TARGET, axis=1), train[TARGET])

    initial_type = [('float_input', FloatTensorType([None, train.drop(TARGET, axis=1).shape[1]]))]
    onx = convert_sklearn(clr, initial_types=initial_type)
    model_name = model.__class__.__name__

    onnx_path = 'e2e_demo/model/artifacts/{}_churn.onnx'.format(model_name)
    yaml_path = 'e2e_demo/model/artifacts/{}_churn.yaml'.format(model_name)

    with open(onnx_path, 'wb') as f:
        f.write(onx.SerializeToString())

    with open(yaml_path, 'w') as f:
        yaml.dump(make_yaml_dict(train, TARGET), f)

    pio_model_list.append(
        (model_name, onnx_path, yaml_path)
    )

p = pio.Project(_id='616d28bee192db001c78566d', name='playground')

experiment = pio.Experiment.from_id('616d2d4ee192db001c785683')

experiment_version = experiment.latest_version

experiment_version.new_version(dataset=train_dset,
                               holdout_dataset=test_dset,
                               target_column=TARGET,
                               external_models=pio_model_list)
