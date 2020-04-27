import random
import sys

if len(sys.argv) != 4:
    print("please provide 3 arguments: dimension of sparse variables, number of non-zero entries, and number of samples")
    sys.exit()

sample_size = eval(sys.argv[3])
sparse_dim = eval(sys.argv[1])
fixed_sprase_size = eval(sys.argv[2])

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
        result_set = set()
        while len(result_set) < fixed_sprase_size:
            result_set.add(random.randint(0, sparse_dim - 1))
        sparse_features = list(result_set)
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
        if random.random() > 0.5:
            label = "1.0"
        f.write(label)
        f.write("|")
        # ts
        f.write("1544094136\n")