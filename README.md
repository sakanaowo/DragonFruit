
# Dragon Fruit Disease Classification

This project focuses on classifying dragon fruit diseases from images using deep learning models.

## 📁 Project Structure

```
.
├── config.yaml                   # Configuration file for parameters
├── data/                         # Dataset split into train/val/test folders
│   ├── train/
│   ├── val/
│   └── test/
├── models/                       # Trained model checkpoints
│   └── temp/
├── notes/                        # Notes and experiment logs
│   └── temp/
├── outputs/                      # Outputs such as images, logs, etc.
│   └── temp/
├── requirements.txt              # Required Python packages
├── Sorted_dataset/               # Pre-sorted and labeled dataset
│   ├── Fungal Infections (Anthracnose or Stem Canker)/
│   ├── Healthy Fruits/
│   ├── Healthy Leaves/
│   ├── Insect-Infected Fruits/
│   ├── Sunburn Damage/
│   ├── labels_fromGPT.json
│   ├── old_label.json
│   ├── sorted_dataset_labels.json
├── src/                          # Main source code
│   ├── dataset.py                # Data preparation and loading
│   ├── evaluate.py               # Model evaluation
│   ├── model.py                  # Model architecture definition
│   ├── predict.py                # Make predictions on new images
│   └── train.py                  # Training script
├── temp.py                       # Temporary script for testing
└── utils/                        # Utility scripts
    ├── config.py                 # Handling configuration
    ├── helper.py                 # Helper functions (save, visual, logs,... )
    ├── split_dataset.py          # Dataset splitting utility
    └── transform.py              # Data transformations
```

## Installation

1. Clone the repository:


```bash
git clone https://github.com/sakanaowo/DragonFruit.git
cd DragonFruit
```

2. Install the required dependencies:


```bash
pip install -r requirements.txt
```

3. Adjust parameters as needed in `config.yaml`.


## Usage

- **Train the model**:


```bash
python src/train.py
```

- **Evaluate the model**:


```bash
python src/evaluate.py
```

- **Predict on a new image**:


```bash
python src/predict.py --image_path <path_to_image>
```

## Dataset

The dataset contains 5 classes:

- Fungal Infections (Anthracnose or Stem Canker)

- Healthy Fruits

- Healthy Leaves

- Insect-Infected Fruits

- Sunburn Damage


The dataset is properly split into training, validation, and testing sets.

## Notes

- The `notes/` folder is used to store experiment logs, observations, and additional notes.

- The JSON files inside `Sorted_dataset/` store class labels and dataset information.

