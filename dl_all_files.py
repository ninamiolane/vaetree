import joblib
import os
from joblib import Parallel, delayed

TARGET = '/neuro/'

def dl_file(path):
    path = path.strip()
    target_path = TARGET + os.path.dirname(path)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    os.system("aws s3 cp s3://openneuro.org/%s %s" % (path, target_path))



def main():
    with open('files') as f:
        all_files = f.readlines()

    Parallel(n_jobs=10)(delayed(dl_file)(f) for f in all_files)


main()
