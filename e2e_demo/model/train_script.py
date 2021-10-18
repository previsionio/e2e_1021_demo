import previsionio as pio
from . import make_yaml_dict
import yaml
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import os
from datetime import datetime

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

clr = make_pipeline(OrdinalEncoder(), DecisionTreeClassifier())
clr.fit(train.drop(TARGET, axis=1), train[TARGET])

initial_type = [('float_input', FloatTensorType([None, train.drop(TARGET, axis=1).shape[1]]))]
onx = convert_sklearn(clr, initial_types=initial_type)

onnx_path = 'e2e_demo/model/artifacts/decision_tree_churn.onnx'
yaml_path = 'e2e_demo/model/artifacts/decision_tree_churn.yaml'

with open(onnx_path, 'wb') as f:
    f.write(onx.SerializeToString())

with open('e2e_demo/model/artifacts/decision_tree_churn.yaml', 'w') as f:
    yaml.dump(make_yaml_dict(train, TARGET), f)

p = pio.Project(_id='616d28bee192db001c78566d', name='playground')

exp = p.create_external_classification(experiment_name='churn_{}'.format(datetime.now().strftime('%Y%m%d_%H%M')),
                                       dataset=train_dset,
                                       holdout_dataset=test_dset,
                                       target_column=TARGET,
                                       external_models=[
                                           ('decision_tree', onnx_path, yaml_path)
                                       ])
