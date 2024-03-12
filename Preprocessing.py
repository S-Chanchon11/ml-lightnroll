import librosa
import os
import json
import numpy as np
from PCP import PitchClassProfiler
from HPCP import my_enhanced_hpcp
import csv

DATASET_PATH = "guitar_chord/Training"
DATA_TEST_PATH = "guitar_chord/Test"
ALL_DATA_PATH = "guitar_chord/All"
# JSON_PATH = "data_maj_chord_v1.json"


SAMPLES_TO_CONSIDER = 44100  # 1 sec. of audio


# Extend the JSONEncoder class
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def preprocess_data_pcp(dataset_path, json_path):
    """Extracts pitch class from music dataset and saves them into a json file along witgh genre labels.
    :param dataset_path (str): Path to dataset
    :param json_path (str): Path to json file used to save pitchs
    :return:
    """

    # dictionary to store mapping, labels, and pitch classes
    data = {"mapping": [], "labels": [], "pitch": [], "order": []}

    # loop through all chord sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        #          C       C#      D      D#     E      F     F#     G      G#     A      A#    B
        # fref_0 = [16.35, 17.32, 18.35, 19.45, 20.60, 21.83, 23.12, 24.50, 25.96, 27.50, 29.14, 30.87]
        # fref_1 = [32.70, 34.65, 36.71, 38.89, 41.20, 43.65, 46.25, 49.00, 51.91, 55.00, 58.27, 61.74]
        # fref = fref_0 + fref_1

        # ensure we're processing a sub-folder level
        if dirpath is not dataset_path:

            # save chord label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in chord sub-dir
            for f in sorted(filenames):

                # load audio file
                file_path = os.path.join(dirpath, f)
                file_name = file_path.split("/")[-1]
                file_name2 = file_name.split(".")[0]
                data["order"].append(file_name2)
                # process all segments of audio file
                # data["pitch"].append(myhpcp(audio_path=file_path, fref=fref_0[i-1]))
                data["pitch"].append(
                    my_enhanced_hpcp(audio_path=file_path, fref=261.63, pcp_num=12)
                )
                data["labels"].append(i - 1)
                # print(int(float(data["order"][0])))
                print("{}, segment:{}".format(file_path, 1))
        else:
            print("oups")

    # save pitch classes to json file
    with open(json_path, "w") as fp:
        # print(type(data))
        json.dump(data, fp, indent=4, cls=NumpyEncoder)


def write_to_csv(dataset_path, csv_path):

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a sub-folder level
        if dirpath is not dataset_path:
            semantic_label = dirpath.split("/")[-1]

            # open the csv file in append mode (a)
            with open(csv_path, "a", newline="") as csv_file:
                writer = csv.writer(csv_file)

                # write header row if the file doesn't exist
                if os.stat(csv_path).st_size == 0:
                    writer.writerow(
                        [
                            "mapping",
                            "order",
                            "pitch_C",
                            "pitch_C#",
                            "pitch_D",
                            "pitch_D#",
                            "pitch_E",
                            "pitch_F",
                            "pitch_F#",
                            "pitch_G",
                            "pitch_G#",
                            "pitch_A",
                            "pitch_A#",
                            "pitch_B",
                            "labels",
                        ]
                    )

                # process all audio files in chord sub-dir
                for f in filenames:
                    file_path = os.path.join(dirpath, f)
                    file_name2 = file_path.split("/")[-1].split(".")[0]

                    pcp_list = my_enhanced_hpcp(
                        audio_path=file_path, fref=261.63, pcp_num=12
                    )

                    # process all segments of audio file
                    data = {
                        "mapping": semantic_label,
                        "order": file_name2,
                        "pitch_C": pcp_list[0],
                        "pitch_C#": pcp_list[1],
                        "pitch_D": pcp_list[2],
                        "pitch_D#": pcp_list[3],
                        "pitch_E": pcp_list[4],
                        "pitch_F": pcp_list[5],
                        "pitch_F#": pcp_list[6],
                        "pitch_G": pcp_list[7],
                        "pitch_G#": pcp_list[8],
                        "pitch_A": pcp_list[9],
                        "pitch_A#": pcp_list[10],
                        "pitch_B": pcp_list[11],
                        "labels": i - 1,
                    }

                    # write data to the csv file
                    writer.writerow(data.values())

                    print("{}, segment:{}".format(file_path, 1))
        else:
            print("oups")


if __name__ == "__main__":

    JSON_PATH = "output/data_all_chord_2.json"
    JSON_PATH_TEST = "output/test/test_all_3.json"

    # preprocess_data_pcp(DATASET_PATH, JSON_PATH)
    write_to_csv(DATASET_PATH, csv_path="output/data_all_chord_3.csv")

    # preprocess_data_pcp(DATA_TEST_PATH, JSON_PATH_TEST)
    # write_to_csv(DATA_TEST_PATH,csv_path="output/output_test_all_3.csv")
