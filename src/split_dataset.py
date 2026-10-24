import pandas as pd 
from pathlib import Path
from sklearn.model_selection import train_test_split 

class Dataset:
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.train_dir = self.output_dir / 'train'
        self.train_dir.mkdir(exist_ok=True, parents=True)
        self.val_dir = self.output_dir / 'val'
        self.val_dir.mkdir(exist_ok=True, parents=True)
    
    def create_splits(self, train_percent: float):
        """
        No test split since we are comparing this against HIPE-OCRepair competition's benchmarks
        """
        for file in self.data_dir.glob("*.parquet"):
            year = file.stem
            article_df = pd.read_parquet(file)
            train_df, val_df = train_test_split(article_df, test_size=(1 - train_percent), random_state=42)
            train_df.to_parquet(self.train_dir / f"{year}_train.parquet")
            val_df.to_parquet(self.val_dir / f"{year}_val.parquet")


if __name__ == "__main__":
    dataset = Dataset("/home/ewong/scratch/distill-ocr/teacher_training_set", "/home/ewong/scratch/distill-ocr/teacher_dataset")
    dataset.create_splits(0.9)

        

        