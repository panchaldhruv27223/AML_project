import os
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def split_dataset(input_csv, output_dir, train_size=4242, test_size_ratio=0.15, include_validation=False):
    """
    Split the dataset into train, test, and optionally validation sets.
    
    Args:
        input_csv (str): Path to the input CSV file.
        output_dir (str): Directory to save the split CSVs.
        train_size (int): Number of sentences for the training set.
        test_size_ratio (float): Proportion of remaining data for test set.
        include_validation (bool): Whether to create a validation set.
    """
    try:
        # Read the raw dataset
        logger.info(f"Reading input CSV: {input_csv}")
        df = pd.read_csv(input_csv)
        total_sentences = len(df)
        logger.info(f"Total sentences in dataset: {total_sentences}")

        # Validate train_size
        if train_size > total_sentences:
            raise ValueError(f"Requested train_size ({train_size}) exceeds total sentences ({total_sentences})")

        # Shuffle the dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split into train and remaining
        train_df = df.iloc[:train_size]
        remaining_df = df.iloc[train_size:]
        remaining_count = len(remaining_df)

        if remaining_count == 0:
            raise ValueError("No sentences left for test/validation after allocating train set")

        # if include_validation:
        #     # Split remaining into test and validation
        #     test_size = int(remaining_count * test_size_ratio)
        #     valid_size = remaining_count - test_size
        #     if test_size < 1 or valid_size < 1:
        #         raise ValueError(f"Remaining {remaining_count} sentences too few to split into test and validation")

        #     test_df, valid_df = train_test_split(
        #         remaining_df,
        #         test_size=valid_size / remaining_count,
        #         random_state=42
        #     )
        # else:
        #     # All remaining go to test
        #     test_df = remaining_df
        #     valid_df = None
        #     test_size = remaining_count

        test_df = df.iloc[train_size: (train_size+500)]

        valid_df = df.iloc[(train_size+500): (train_size+1000)]

        # Log split sizes
        logger.info(f"Train set size: {len(train_df)} sentences")
        logger.info(f"Test set size: {len(test_df)} sentences")
        if include_validation:
            logger.info(f"Validation set size: {len(valid_df)} sentences")

        # Save to CSVs
        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, "train_split.csv")
        test_path = os.path.join(output_dir, "test_split.csv")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        logger.info(f"Saved train split to {train_path}")
        logger.info(f"Saved test split to {test_path}")

        if include_validation:
            valid_path = os.path.join(output_dir, "valid_split.csv")
            valid_df.to_csv(valid_path, index=False)
            logger.info(f"Saved validation split to {valid_path}")

    except Exception as e:
        logger.error(f"Error during dataset splitting: {str(e)}")
        raise

def main():
    # Paths
    base_path = "D:\Manish Prajapati\LibriSpeech\csv"
    input_csv = os.path.join(base_path, "train.csv")  # Adjust this to your raw CSV filename
    output_dir = base_path

    # Split the dataset
    split_dataset(
        input_csv=input_csv,
        output_dir=output_dir,
        train_size=4242,  # As requested
        test_size_ratio=0.15,  # 15% of remaining for test
        include_validation=False  # Set to True if you want a validation set
    )

if __name__ == "__main__":
    main()