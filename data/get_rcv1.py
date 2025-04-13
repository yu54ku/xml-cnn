import os
import subprocess
import argparse
import requests
from tqdm import tqdm
import gzip
import shutil

label_dict = {}
url = "http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/"
label_url = "http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a08-topic-qrels/"
files = [
    ("lyrl2004_tokens_train.dat", 5108963),
    ("lyrl2004_tokens_test_pt0.dat", 44734992),
    ("lyrl2004_tokens_test_pt1.dat", 45595102),
    ("lyrl2004_tokens_test_pt2.dat", 44507510),
    ("lyrl2004_tokens_test_pt3.dat", 42052117),
]
label_file = "rcv1-v2.topics.qrels"

def get_num_of_doc(path):
    """Count number of documents in file by counting '.W' markers"""
    count = 0
    with open(path, 'r') as f:
        for line in f:
            if '.W' in line:
                count += 1
    return count

def download_file(url, filename, filesize, skip_if_exists=False):
    """Download a file with progress bar"""
    if skip_if_exists and os.path.exists(filename):
        print(f"File {filename} already exists, skipping download...")
        return True
    
    if skip_if_exists and os.path.exists(filename + ".gz"):
        print(f"File {filename}.gz already exists, skipping download...")
        return True
        
    try:
        with open(filename + ".gz", "wb") as file:
            pbar = tqdm(total=filesize, unit="B", unit_scale=True)
            pbar.set_description(f"Downloading {filename[16:]}.gz")
            response = requests.get(url + filename + ".gz", stream=True)
            if response.status_code != 200:
                print(f"Failed to download {filename}.gz. Status code: {response.status_code}")
                return False
                
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
                pbar.update(len(chunk))
            pbar.close()
        return True
    except Exception as e:
        print(f"Error downloading {filename}.gz: {str(e)}")
        return False

def extract_file(filename, remove_gz=True):
    """Extract gzipped file"""
    if not os.path.exists(filename + ".gz"):
        print(f"Warning: {filename}.gz does not exist, cannot extract")
        return False
        
    if os.path.exists(filename):
        print(f"File {filename} already exists, skipping extraction...")
        return True
        
    try:
        with gzip.open(filename + ".gz", 'rb') as f_in:
            with open(filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        if remove_gz:
            os.remove(filename + ".gz")
        return True
    except Exception as e:
        print(f"Error extracting {filename}.gz: {str(e)}")
        return False

def get_labels(doc_id, label_dict):
    """Get labels for a document ID"""
    labels = label_dict.get(doc_id, [])
    return " ".join(labels)

def process_document_file(filename, label_dict):
    """Process a single document file"""
    if not os.path.exists(filename):
        print(f"Error: {filename} does not exist, cannot process")
        return False
        
    if os.path.exists(filename + ".out"):
        print(f"Output file {filename}.out already exists, skipping processing...")
        return True
    
    try:
        num_of_doc = get_num_of_doc(filename)
        
        with open(filename) as f:
            flag = False
            buf = []
            doc_id = []
            datafile = tqdm(f, total=num_of_doc, unit="Docs")
            datafile.set_description(f"Processing {filename[16:]}")
            
            for i in f:
                if (".I" in i) and (not flag):
                    doc_id.append(i.replace(".I ", "")[:-1])
                    flag = True
                elif ".I" in i:
                    doc_id.append(i.replace(".I ", "")[:-1])
                    # Process the document here
                    labels = get_labels(doc_id[-2], label_dict)
                    text = " ".join(buf).replace("\n", "").replace(".W", "")
                    
                    output = doc_id[-2] + "\t" + labels + "\t" + text[1:-1] + "\n"
                    
                    with open(filename + ".out", "a") as f_output:
                        f_output.write(output)
                        buf = []
                    datafile.update()
                else:
                    buf.append(i)
            else:
                labels = get_labels(doc_id[-1], label_dict)
                text = " ".join(buf).replace("\n", "").replace(".W", "") + "\n"
                
                output = doc_id[-1] + "\t" + labels + "\t" + text
                
                with open(filename + ".out", "a") as f_output:
                    f_output.write(output)
                    buf = []
                
                datafile.update()
        
        datafile.close()
        os.remove(filename)
        return True
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return False

def process_label_file(label_file):
    """Process the label file"""
    if not os.path.exists(label_file):
        print(f"Error: {label_file} does not exist, cannot process labels")
        return False
    
    try:
        with open(label_file) as f:
            print("Processing label file...")
            for line in tqdm(f, total=2606875, desc="Processing labels"):
                parts = line.strip().split()
                if len(parts) >= 3:
                    category = parts[0]  
                    doc_id = parts[1]    
                    
                    if doc_id not in label_dict:
                        label_dict[doc_id] = []
                    
                    label_dict[doc_id].append(category)
        
        os.remove(label_file)
        print(f"Processed labels for {len(label_dict)} documents")
        return True
    except Exception as e:
        print(f"Error processing {label_file}: {str(e)}")
        return False

def combine_files(output_files):
    """Combine processed files into train and test files"""
    try:
        if os.path.exists(output_files[0]):
            if os.name == 'nt': 
                shutil.copy(output_files[0], "train_org.txt")
                os.remove(output_files[0])
            else: 
                subprocess.run(["mv", output_files[0], "train_org.txt"], stdout=subprocess.PIPE)
        else:
            print(f"Warning: {output_files[0]} not found, cannot create train_org.txt")
            return False
        
        if os.name == 'nt':  
            with open("test.txt", 'w') as outfile:
                for fname in output_files[1:]:
                    if os.path.exists(fname):
                        with open(fname) as infile:
                            outfile.write(infile.read())
                        os.remove(fname)
                    else:
                        print(f"Warning: {fname} not found, continuing...")
        else:  
            cmd = ["cat " + " ".join([f for f in output_files[1:] if os.path.exists(f)]) + " > test.txt"]
            subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
            [os.remove(f) for f in output_files[1:] if os.path.exists(f)]
        
        return True
    except Exception as e:
        print(f"Error combining files: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download and process RCV1 dataset with resumable steps')
    parser.add_argument('--skip-download', action='store_true', help='Skip downloading files if they exist')
    parser.add_argument('--skip-extraction', action='store_true', help='Skip extracting gz files')
    parser.add_argument('--skip-label-processing', action='store_true', help='Skip label file processing')
    parser.add_argument('--start-from-processing', action='store_true', help='Start from processing documents (skip download and extraction)')
    parser.add_argument('--process-labels-first', action='store_true', help='Process labels before processing documents')
    
    args = parser.parse_args()
    
    print("RCV1 Downloader with resumable steps")
    print("This program downloads files from '" + url[:-1] + "'.")
    
    if args.process_labels_first and not args.skip_label_processing:
        print("\n--- Processing Labels ---")
        if not args.start_from_processing:
            if download_file(label_url, label_file, 7272130, args.skip_download):
                if not args.skip_extraction:
                    extract_file(label_file)
        
        if os.path.exists(label_file):
            process_label_file(label_file)
        else:
            print("Warning: Label file not found, continuing without labels")
    
    print("\n--- Processing Documents ---")
    output_files = []
    
    for filename, filesize in files:
        if not args.start_from_processing:
            if not download_file(url, filename, filesize, args.skip_download):
                print(f"Skipping {filename} due to download failure")
                continue
            
            if not args.skip_extraction:
                if not extract_file(filename):
                    print(f"Skipping {filename} due to extraction failure")
                    continue
        
        if process_document_file(filename, label_dict):
            output_files.append(filename + ".out")
    
    if not args.process_labels_first and not args.skip_label_processing:
        print("\n--- Processing Labels ---")
        if not args.start_from_processing:
            if download_file(label_url, label_file, 7272130, args.skip_download):
                if not args.skip_extraction:
                    extract_file(label_file)
        
        if os.path.exists(label_file):
            process_label_file(label_file)
        else:
            print("Warning: Label file not found, continuing without labels")
    
    print("\n--- Combining Files ---")
    combine_files([f for f in output_files if os.path.exists(f)])
    
    print("\nAll processing complete!")

if __name__ == "__main__":
    main()