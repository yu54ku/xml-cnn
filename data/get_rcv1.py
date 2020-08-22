# RCV1 Downloader (There's no guarantee that this program will work.)

import os
import subprocess

import numpy as np
import requests
from sklearn.datasets import fetch_rcv1
from tqdm import tqdm


def get_num_of_doc(path):
    cmd = "cat " + path + " | grep .W | wc -l "
    output = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout
    return int(output.decode("utf8").split(" ")[0])


url = "http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/"

files = [
    ("lyrl2004_tokens_train.dat", 5108963),
    ("lyrl2004_tokens_test_pt0.dat", 44734992),
    ("lyrl2004_tokens_test_pt1.dat", 45595102),
    ("lyrl2004_tokens_test_pt2.dat", 44507510),
    ("lyrl2004_tokens_test_pt3.dat", 42052117),
]

print("This program downloads files from '" + url[:-1] + "'.")

print("\nLoad RCV1 labels from sklearn (a few minutes)...  ", end="", flush=True)
rcv1 = fetch_rcv1()
sample_id = rcv1.sample_id
target_names = rcv1.target_names.tolist()
target = rcv1.target
print("Done.\n", flush=True)


for filename, filesize in files:
    with open(filename + ".gz", "wb") as file:
        pbar = tqdm(total=filesize, unit="B", unit_scale=True)
        pbar.set_description("Downloading " + filename[16:] + ".gz")
        for chunk in requests.get(url + filename + ".gz", stream=True).iter_content(
            chunk_size=1024
        ):
            ff = file.write(chunk)
            pbar.update(len(chunk))
        pbar.close()

    cmd = ["gzip", "-d", filename + ".gz"]
    subprocess.run(cmd, stdout=subprocess.PIPE)

    num_of_doc = get_num_of_doc(filename)

    with open(filename) as f:
        flag = False
        buf = []
        doc_id = []
        datafile = tqdm(f, total=num_of_doc, unit="Docs")
        datafile.set_description("Processing " + filename[16:])
        for i in f:
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
                datafile.update()
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

            datafile.update()

    datafile.close()
    os.remove(filename)

files = [i[0] + ".out" for i in files]

cmd = ["mv", files[0], "train_org.txt"]
subprocess.run(cmd, stdout=subprocess.PIPE)

cmd = ["cat " + " ".join(files[1:]) + " > test.txt"]
subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
[os.remove(i) for i in files[1:]]
