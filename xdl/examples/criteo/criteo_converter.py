import sys
import csv

NUM_INTEGER_FEATURES = 13
NUM_CATEGORICAL_FEATURES = 26

if len(sys.argv) != 3:
    print("please provide 2 arguments: file path, number of columns")
    sys.exit()

file_path = sys.argv[1]
num_columns = eval(sys.argv[2])

# pass one: build map
map_cat_feature_to_int = {}
feature_counts = [0] * NUM_CATEGORICAL_FEATURES

with open(file_path) as fcsv:
    count = 0
    for line in csv.reader(fcsv, dialect="excel-tab"):
        count += 1
        for i in range(0, NUM_CATEGORICAL_FEATURES):
            feature = line[i + NUM_INTEGER_FEATURES+1]
            if feature not in map_cat_feature_to_int:
                map_cat_feature_to_int[feature] = feature_counts[i]
                feature_counts[i] += 1
        if count >= num_columns:
            break
# iterative sum
intervals = []
total = 0
for x in feature_counts:
    intervals.append(total)
    total += x

print("sparse dimension: " + str(total))

# pass two: write new file
with open(file_path) as fcsv:
    with open('generated_data.txt', 'w') as f:
        count = 0
        for line in csv.reader(fcsv, dialect="excel-tab"):
            # sample id
            f.write("s" + str(count))
            f.write("|")

            # group id
            f.write("g" + str(count))
            f.write("|")

            # sparse data
            f.write("sparse0@")
            for i in range(0, NUM_CATEGORICAL_FEATURES):
                if i > 0:
                    f.write(",")
                feature = line[i + NUM_INTEGER_FEATURES + 1]
                sparse_id = map_cat_feature_to_int[feature] + intervals[i]
                f.write(str(sparse_id) + ":1.0")
            f.write("|")

            # dense data
            f.write("dense0@")
            for i in range(0, NUM_INTEGER_FEATURES):
                if i > 0:
                    f.write(",")
                feature = line[i + 1]
                f.write(feature)

            # label
            f.write(line[0])
            f.write("|")
            # ts
            f.write("1544094136\n")





            count += 1
            if count >= num_columns:
                break