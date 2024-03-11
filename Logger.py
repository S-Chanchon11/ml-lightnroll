import csv
import os


class Logger:

    def __init__(self, path):
        self.path = path

    def history(self, filename, target, predict):

        with open(self.path, "a", newline="") as csv_file:
            writer = csv.writer(csv_file)

            if os.stat(self.path).st_size == 0:
                writer.writerow(["filename", "target", "predict"])

            data = {"filename": filename, "target": target, "predict": predict}
