import requests
import gzip
import shutil
import csv
import os
import time

NUM_DAYS = 23
DATA_ZIP_PATH = 'data.zip'
DATA_PATH = 'data.txt'

NUM_INTEGER_FEATURES = 13
NUM_CATEGORICAL_FEATURES = 26

entry_count = 0
feature_usage_count = {}

def reportFeatureUsage():
    count_map = {}
    with open("/users/kaigel/output.txt", "a") as f:
        f.write("total number of features: " + str(len(feature_usage_count)) + "\n")
        for feature in feature_usage_count:
            count = feature_usage_count[feature]
            if count in count_map:
                count_map[count] += 1
            else:
                count_map[count] = 1
        keys = sorted(count_map.keys())
        f.write("frequency report:\n")
        for k in keys:
            f.write(str(k) + ": " + str(count_map[k]) + "\n")

for day in range(NUM_DAYS):
    print("Processing day " + str(day))
    url = 'http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_' + str(day) + '.gz'
    print(url)
    timeBeforeDownload = time.time()
    success = False
    while not success:
        print("Before getting")
        request = requests.get(url, timeout=10, stream=True)
        print("Finished getting")
        success = True
        with open(DATA_ZIP_PATH, 'wb') as f:
            try:
                chunkCount = 0
                for chunk in request.iter_content(1024 * 1024):
                    if chunkCount % 1000 == 0:
                        print("Finished downloading chunk " + str(chunkCount))
                    f.write(chunk)
                    chunkCount += 1
            except Exception as e:
                print("failed")
                success = False
                os.remove(DATA_ZIP_PATH)

    timeAfterDownload = time.time()
    print("Download takes " + str(timeAfterDownload - timeBeforeDownload))
    print("Finished retrieving data for day " + str(day))
    with gzip.open(DATA_ZIP_PATH, 'rb') as f_in:
        with open(DATA_PATH, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print("Finished unzipping data for day " + str(day))
    with open(DATA_PATH) as fcsv:
        for line in csv.reader(fcsv, dialect="excel-tab"):
            if entry_count % 1000000 == 0:
                print("Analyzing entry count " + str(entry_count) + " for day " + str(day))
            entry_count += 1
            for i in range(0, NUM_CATEGORICAL_FEATURES):
                feature = line[i + NUM_INTEGER_FEATURES+1]
                if feature not in feature_usage_count:
                    feature_usage_count[feature] = 1
                else:
                    feature_usage_count[feature] += 1
    print("reporting for day " + str(day))
    reportFeatureUsage()
    print("removing zipped data")
    os.remove(DATA_ZIP_PATH)
    print("Removing data")
    os.remove(DATA_PATH)
    print("All files Removed!")