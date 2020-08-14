import sys
import csv
from heapq import heappush, heappop

NUM_INTEGER_FEATURES = 13
NUM_CATEGORICAL_FEATURES = 26

if len(sys.argv) != 4:
    print("please provide 3 arguments: file path, number of columns read, and number of columns written")
    sys.exit()

file_path = sys.argv[1]
num_columns_read = eval(sys.argv[2])
num_columns_written = eval(sys.argv[2])

# pass one: build map
map_cat_feature_to_freq = {}

with open(file_path) as fcsv:
    count = 0
    for line in csv.reader(fcsv, dialect="excel-tab"):
        count += 1
        for i in range(0, NUM_CATEGORICAL_FEATURES):
            feature = line[i + NUM_INTEGER_FEATURES+1]
            if feature not in map_cat_feature_to_freq:
                map_cat_feature_to_freq[feature] = 1
            else:
                map_cat_feature_to_freq[feature] += 1
        if count >= num_columns_read:
            break
# create heap
h = []
for key, val in map_cat_feature_to_freq.items():
    heappush(h, (val, key))

map_cat_feature_to_int = {}
feature_count = 0
while len(h) > 0:
    val = heappop(h)[1]
    map_cat_feature_to_int[val] = feature_count
    feature_count += 1
print(map_cat_feature_to_int)

print("sparse dimension: " + str(feature_count))

# pass two: write new file
with open(file_path) as fcsv:
    with open('generated_data.txt', 'w') as f:
        line_num = 0
        for line in csv.reader(fcsv, dialect="excel-tab"):
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
                feature = line[i + NUM_INTEGER_FEATURES + 1]
                sparse_id = map_cat_feature_to_int[feature]
                f.write(str(sparse_id) + ":1.0")
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
            if line_num >= num_columns_written:
                break
