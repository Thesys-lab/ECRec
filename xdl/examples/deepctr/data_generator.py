import random
sample_size = 10000
sparse_dim = 4096
fixed_sprase_size = 10
label_1_threashold = 400
# if present dimension 0 - 400 present, will match to label 1.0
with open('generated_data.txt', 'w') as f:
    for line_num in range(sample_size):
        # sample id
        f.write("s" + str(line_num))
        f.write("|")
        # group id
        f.write("g" + str(line_num))
        f.write("|")
        # sparse data
        f.write("sparse0@")
        sparse_features = random.sample(range(4096), fixed_sprase_size)
        for i in range(len(sparse_features)):
            if i > 0:
                f.write(",")
            feature = sparse_features[i]
            f.write(str(feature) + ":1.0")
        f.write("|")
        # no dense feature
        f.write("|")
        # label
        label = "0.0"
        for k in sparse_features:
            if k < label_1_threashold:
                label = "1.0"
                break
        f.write(label)
        print(label)
        f.write("|")
        # ts
        f.write("1544094136\n")