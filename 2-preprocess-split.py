import csv
import random
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

def split_csv(input_csv, output_dir, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):
    """Split input CSV into train, dev, test CSVs."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Read input CSV
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        lines = list(reader)
    
    if len(lines) < 2:
        logger.error(f"Input CSV {input_csv} is empty or invalid")
        return
    
    # Extract header and data
    header = lines[0]
    data = lines[1:]
    
    # Shuffle data
    random.seed(42)  # For reproducibility
    random.shuffle(data)
    
    # Calculate split sizes
    total_lines = len(data)
    train_size = int(total_lines * train_ratio)
    dev_size = int(total_lines * dev_ratio)
    test_size = total_lines - train_size - dev_size  # Ensure all lines are used
    
    # Split data
    train_data = data[:train_size]
    dev_data = data[train_size:train_size + dev_size]
    test_data = data[train_size + dev_size:]
    
    # Write output CSVs
    for split_name, split_data in [("train_split", train_data), ("dev_split", dev_data), ("test_split", test_data)]:
        output_csv = os.path.join(output_dir, f"{split_name}.csv")
        with open(output_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header)
            writer.writerows(split_data)
        logger.info(f"Created {output_csv} with {len(split_data)} samples")

if __name__ == "__main__":
    input_csv = r"D:\Manish Prajapati\LibriSpeech\csv\train.csv"
    output_dir = r"D:\Manish Prajapati\LibriSpeech\csv"
    split_csv(input_csv, output_dir)