import pickle

from ModelPrediction import init_data
from Preprocessing import preprocess_data_pcp
from Utilities import Utilities

DATA_TEST_PATH = "guitar_chord/Test"
JSON_TEST_PATH = "test_multi.json"

def main():

    utils = Utilities()

    # preprocess_data_pcp(DATA_TEST_PATH,JSON_TEST_PATH)

    X_test,y_test,z_test = utils.prepare_data(path=JSON_TEST_PATH)

    mapping=[
        "Am",
        "Em",
        "G",
        "A",
        "F",
        "Dm",
        "C",
        "D",
        "E",
        "B"
    ]

    ann1 = 'output/model/ann_i.h5'
    ann2 = 'output/model/ann_ii.h5'
    ann3 = 'output/model/ann_iii.h5'

    load_ann1 = pickle.load(open(ann1, 'rb')) 
    load_ann2 = pickle.load(open(ann2, 'rb')) 
    load_ann3 = pickle.load(open(ann3, 'rb')) 

    ann1_pred = load_ann1.predict(X_test)
    ann2_pred = load_ann2.predict(X_test)
    ann3_pred = load_ann3.predict(X_test)

    for i in range(len(y_test)):

        print(mapping[ann1_pred[i]],end=' ')
        print(mapping[ann2_pred[i]],end=' ')
        print(mapping[ann3_pred[i]])

# what if every model answer differently?
        

if __name__ == '__main__':
    main()