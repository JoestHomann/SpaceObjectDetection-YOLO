from pathlib import Path

# File paths as variables
dataset_path = Path("D:/SpaceObjectDetection-YOLO/dataset")      # Spark 2022 Stream 1 Dataset Directory
labels_path = dataset_path / "labels"                            # Labels Directory

train_csv_path = labels_path / "train.csv"                       # Training Dataset Labels CSV File
val_csv_path = labels_path / "val.csv"                           # Validation Dataset Labels CSV File

train_img_path = dataset_path / "train" / "train"                # Training Images Directory
val_img_path = dataset_path / "val" / "val"                      # Validation Images Directory

output_path = labels_path / "labels_yolo"                        # Output Directory For YOLO Formatted Labels
output_train_path = output_path / "train"                        # Output Directory For Training YOLO Labels
output_val_path = output_path / "val"                            # Output Directory For Validation YOLO Labels



def main():

    # Resolve Dataset Path
    print("Dataset Path:", dataset_path.resolve())

    # Verify Existence Of Paths
    print("Training CSV Path:", train_csv_path, "Exists:", train_csv_path.exists())
    print("Validation CSV Path:", val_csv_path, "Exists:", val_csv_path.exists())
    print("Training Images Path:", train_img_path, "Exists:", train_img_path.exists())
    print("Validation Images Path:", val_img_path, "Exists:", val_img_path.exists())

    # Resolve Output Path
    print("Output Path:", output_path.resolve())

    # Verify Existence Of Output Paths
    print("Output Training Path:", output_train_path, "Exists:", output_train_path.exists())
    print("Output Validation Path:", output_val_path, "Exists:", output_val_path.exists())


if __name__ == "__main__":
    main()

