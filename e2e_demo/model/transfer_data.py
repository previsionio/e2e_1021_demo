import previsionio as pio
import pandas as pd
import yaml
import os

pio.client.init_client(token=os.environ['PIO_MASTER_TOKEN'],
                       prevision_url='https://pistre.prevision.io')

train = pd.read_csv('gs://prevision-public/churn_train_vivatech.csv')
test = pd.read_csv('gs://prevision-public/churn_test_vivatech.csv')

TARGET = 'contract_ended'

train[TARGET] = train[TARGET].astype(int)
test[TARGET] = test[TARGET].astype(int)

train = train.loc[:, [c for c, t in train.dtypes.items() if t == 'float64' or c == TARGET]]
test = test.loc[:, [c for c, t in test.dtypes.items() if t == 'float64' or c == TARGET]]

p = pio.Project(_id='616d28bee192db001c78566d', name='playground')

train_dset = p.create_dataset(name='churn_train', dataframe=train)
holdout_dset = p.create_dataset(name='churn_holdout', dataframe=test)

dset_dict = {'train': train_dset.id, 'test': holdout_dset.id}

with open('e2e_demo/model/artifacts/dset_config.yaml', 'w') as f:
    yaml.dump(dset_dict, f)
