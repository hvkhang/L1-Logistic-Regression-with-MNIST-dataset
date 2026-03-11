# MNIST L1 Logistic Regression

This repository contains a Jupyter notebook demonstrating binary classification of the MNIST dataset using logistic regression with L1 regularization. The focus is on distinguishing digits 0 and 8, and exploring how sparsity introduced by the L1 penalty impacts model performance.

## Files

- `main.ipynb` – Main notebook that:
  - Imports necessary libraries
  - Loads the MNIST dataset from OpenML
  - Filters and samples digits 0 and 8
  - Preprocesses and splits the data
  - Trains an L1-regularized logistic regression model
  - Evaluates and visualizes results (accuracy, confusion matrix, sample images)

## Dataset

The notebook uses the `mnist_784` dataset from [OpenML](https://www.openml.org/d/554). It contains 70,000 28×28 grayscale images of handwritten digits (0–9). For simplicity, only samples of digits 0 and 8 are used, with 3,000 images per class.

## Usage

1. **Install dependencies** (e.g. create a virtual environment and run):
   ```bash
   pip install -r requirements.txt
   ```
2. **Open the notebook** in Jupyter or VS Code:
   ```bash
   jupyter notebook main.ipynb
   ```
3. **Run cells sequentially** to reproduce the analysis. The dataset is downloaded automatically by scikit-learn on first run.

## Notes

- The notebook demonstrates data filtering, sampling, PCA visualization, and model training with `sklearn.linear_model.LogisticRegression` (solver `liblinear`, penalty `l1`).
- Labels are transformed to `-1` and `1` to match the L1 logistic regression implementation.

Feel free to modify the notebook to explore other digits, regularization strengths, or classification techniques."# L1-Logistic-Regression-with-MNIST-dataset" 

## Author
- **Vuong Khang Huynh** – Ho Chi Minh City University of Technology - Faculty of Computer Science and Engineering.  
- **Huu Loi Bui** - Ho Chi Minh City University of Technology - Faculty of Computer Science and Engineering.
