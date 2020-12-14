import pandas as pd
from src.individual import Individual


INPUT_FEATURES = "./data/WineQualityPreparedCleanAttributes_2.csv"
OUTPUT_CLASS = "./data/WineQualityOneHotEncodedClasses.csv"

features = pd.read_csv(INPUT_FEATURES, sep=',')
features.head()

classes = pd.read_csv(OUTPUT_CLASS, sep=',')
classes.head()

TRAIN_RATE = 0.8
VALIDATION_RATE = 0.5
TESTING_RATE = 1 - VALIDATION_RATE

n_instances = features.shape[0]
n_train = int(n_instances * TRAIN_RATE)
n_validation = int((n_instances - n_train) * VALIDATION_RATE)

x_train = features.values[:n_train]
x_validation = features.values[n_train:n_train+n_validation]
x_test = features.values[n_train+n_validation:]

y_train = classes.values[:n_train]
y_validation = classes.values[n_train:n_train+n_validation]
y_test = classes.values[n_train+n_validation:]

print('Train shape: ', x_train.shape, y_train.shape)
print('Validation shape: ', x_validation.shape, y_validation.shape)
print('Test shape: ', x_test.shape, y_test.shape)

nn = Individual(0.9, 0.1)
nn.random_individual()
print(nn.structure)

nn.train(x_train, y_train, x_validation, y_validation)

print(nn.__dict__)