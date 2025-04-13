import argparse
import os
import sys
import shutil

label_dict = {}

def load_labels(label_file):
    """Load labels from RCV1 qrels file"""
    if not os.path.exists(label_file):
        print(f"Error: Label file {label_file} does not exist")
        return False
        
    print(f"Loading labels from {label_file}...")
    count = 0
    
    try:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    category = parts[0]
                    doc_id = parts[1]
                    
                    if doc_id not in label_dict:
                        label_dict[doc_id] = []
                        
                    label_dict[doc_id].append(category)
                    count += 1
                    
                    if count % 100000 == 0:
                        print(f"Processed {count} labels...")
        
        print(f"Successfully loaded {count} labels for {len(label_dict)} documents")
        
        samples = list(label_dict.items())[:3]
        for doc_id, labels in samples:
            print(f"Sample - Doc ID: {doc_id}, Labels: {labels}")
            
        return True
    except Exception as e:
        print(f"Error loading labels: {e}")
        return False

def create_labeled_file(input_file, output_file):
    """Create a new file with labels inserted"""
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist")
        return False
        
    print(f"Processing {input_file} -> {output_file}")
    line_count = 0
    
    try:
        with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue  
                    
                parts = line.split(None, 1)
                if len(parts) >= 1:
                    doc_id = parts[0].strip()
                    
                    text = ""
                    if len(parts) > 1:
                        text = parts[1].strip()
                        
                    labels = label_dict.get(doc_id, [])
                    labels_str = " ".join(labels)
                    
                    fout.write(f"{doc_id}\t{labels_str}\t{text}\n")
                    line_count += 1
                else:
                    fout.write(line + "\n")
                    print(f"Warning: Could not parse line: {line}")
        
        print(f"Processed {line_count} lines in {input_file}")
        return True
    except Exception as e:
        print(f"Error processing file {input_file}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Create new files with labels inserted")
    parser.add_argument("label_file", help="Path to the RCV1 label file (qrels)")
    parser.add_argument("--files", nargs='+', default=["train.txt", "valid.txt", "test.txt"], 
                       help="Files to process (default: train.txt valid.txt test.txt)")
    parser.add_argument("--keep-originals", action="store_true", 
                       help="Keep original files (don't replace them)")
    
    args = parser.parse_args()
    
    if not load_labels(args.label_file):
        print("Failed to load labels. Exiting.")
        return 1
        
    success = True
    for file_path in args.files:
        temp_file = file_path + ".new"
        
        if create_labeled_file(file_path, temp_file):
            if not args.keep_originals:
                backup_file = file_path + ".bak"
                try:
                    shutil.copy2(file_path, backup_file)
                    print(f"Created backup of {file_path} at {backup_file}")
                    
                    shutil.move(temp_file, file_path)
                    print(f"Replaced {file_path} with labeled version")
                except Exception as e:
                    print(f"Error replacing file: {e}")
                    success = False
        else:
            print(f"Failed to process {file_path}")
            success = False
            
    if success:
        print("\nAll files processed successfully!")
    else:
        print("\nSome files could not be processed correctly.")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())