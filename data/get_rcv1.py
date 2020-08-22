import os
import subprocess
import sys

import numpy as np
from sklearn.datasets import fetch_rcv1
from tqdm import tqdm


def get_num_of_line(path):
    if path is None:
        return None
    else:
        cmd = ["wc", "-l", path]
        output = subprocess.run(cmd, stdout=subprocess.PIPE).stdout
        return int(output.decode("utf8").split(" ")[0])


files = [
    "lyrl2004_tokens_train.dat",
    "lyrl2004_tokens_test_pt0.dat",
    "lyrl2004_tokens_test_pt1.dat",
    "lyrl2004_tokens_test_pt2.dat",
    "lyrl2004_tokens_test_pt3.dat",
]


url = "http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/"

for filename in files:
    cmd = ["wget", url + filename + ".gz"]
    subprocess.run(cmd, stdout=subprocess.PIPE)

    cmd = ["gzip", "-d", filename + ".gz"]
    subprocess.run(cmd, stdout=subprocess.PIPE)

    rcv1 = fetch_rcv1()
    sample_id = rcv1.sample_id
    target_names = rcv1.target_names.tolist()
    target = rcv1.target

    num_of_line = get_num_of_line(filename)

    with open(filename) as f:
        flag = False
        buf = []
        doc_id = []
        for i in tqdm(f, total=num_of_line):
            if (".I" in i) and (not flag):
                doc_id.append(i.replace(".I ", "")[:-1])
                flag = True
            elif ".I" in i:
                doc_id.append(i.replace(".I ", "")[:-1])
                index = np.where(sample_id == int(doc_id[-2]))[0]

                labels_bool = np.array(target[index].toarray()[0])
                labels = [target_names[int(i)] for i in np.where(labels_bool == 1)[0]]
                labels = " ".join(labels)
                text = " ".join(buf).replace("\n", "").replace(".W", "")

                output = doc_id[-2] + "\t" + labels + "\t" + text[1:-1] + "\n"

                with open(filename + ".out", "a") as f_output:
                    f_output.write(output)
                    buf = []
            else:
                buf.append(i)
        else:
            index = np.where(sample_id == int(doc_id[-1]))[0]

            labels_bool = np.array(target[index].toarray()[0])
            labels = [target_names[int(i)] for i in np.where(labels_bool == 1)[0]]
            labels = " ".join(labels)
            text = " ".join(buf).replace("\n", "").replace(".W", "") + "\n"

            output = doc_id[-1] + "\t" + labels + "\t" + text

            with open(filename + ".out", "a") as f_output:
                f_output.write(output)
                buf = []

    os.remove(filename)

files = [i + ".out" for i in files]

cmd = ["mv", files[0], "train_org.txt"]
subprocess.run(cmd, stdout=subprocess.PIPE)

cmd = ["cat " + " ".join(files[1:]) + " > test.txt"]
subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
[os.remove(i) for i in files[1:]]
