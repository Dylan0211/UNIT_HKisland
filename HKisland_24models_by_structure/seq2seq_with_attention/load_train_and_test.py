from HKisland_24models_by_structure.lstm.read_data import load_data
from HKisland_24models_by_structure.lstm.train import train_model
from HKisland_24models_by_structure.lstm.test import test_model


if __name__ == '__main__':
    load_data()
    train_model()
    test_model()
