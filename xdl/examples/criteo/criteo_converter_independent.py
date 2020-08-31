import gzip
import shutil
import csv
import time
import requests
import os
import sys

NUM_INTEGER_FEATURES = 13
NUM_CATEGORICAL_FEATURES = 26
sparse_dims = [39884406,39043,17289,7420,20263,3,7120,1543,63,38532951,2953546,403346,10,2208,11938,155,4,976,14,39979771,25641295,39664984,585935,12972,108,36]

accumulative = []

start_day = 0
end_day = 24

embedding_table_dim = 0
for x in sparse_dims:
    accumulative.append(embedding_table_dim)
    embedding_table_dim += x

file_path = []

def id_conv(sparse_id, sparse_index):
    return accumulative[sparse_index] + hash(sparse_id) % sparse_dims[sparse_index]


for day in range(start_day, end_day):
    print("Processing day " + str(day))
    url = 'http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_' + str(day) + '.gz'
    print(url)
    timeBeforeDownload = time.time()
    success = False
    gzip_path = "day_" + str(day) + ".gz"
    unzipped_path = "day_" + str(day)
    processed_path = "day_" + str(day) + "_processed"

    while not success:
        print("Before getting")
        request = requests.get(url, timeout=10, stream=True)
        print("Finished getting")
        success = True
        with open(gzip_path, 'wb') as f:
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
                os.remove(gzip_path)

    timeAfterDownload = time.time()
    print("Download takes " + str(timeAfterDownload - timeBeforeDownload))
    print("Finished retrieving data for day " + str(day))
    with gzip.open(gzip_path, 'rb') as f_in:
        with open(unzipped_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print("Finished unzipping data for day " + str(day))
    os.remove(gzip_path)

    with open(unzipped_path) as fcsv:
        with open(processed_path, 'w') as f:
            line_num = 0
            for line in csv.reader(fcsv, dialect="excel-tab"):
                if (line_num % 100000 == 0):
                    print("Processed line " + str(line_num))
                # sample id
                f.write("s" + str(line_num))
                f.write("|")
                # group id
                f.write("g" + str(line_num))
                f.write("|")
                # sparse data
                f.write("sparse0@")
                for i in range(0, NUM_CATEGORICAL_FEATURES):
                    if i > 0:
                        f.write(",")
                    sparse_id = line[i + NUM_INTEGER_FEATURES + 1]
                    r = id_conv(sparse_id, i)
                    f.write(str(r) + ":1.0")
                f.write("|")

                # dense data
                f.write("dense0@")
                for i in range(0, NUM_INTEGER_FEATURES):
                    if i > 0:
                        f.write(",")
                    feature = line[i + 1]
                    if len(feature) > 0:
                        f.write(feature)
                    else:
                        f.write('0')
                f.write("|")
                # label
                f.write(line[0])
                f.write("|")
                # ts
                f.write("1544094136\n")

                line_num += 1
    os.remove(unzipped_path)
    os.system("sudo aws s3 cp " + processed_path + " s3://criteo-terabytes")
    os.remove(processed_path)

