# Program Information
This project runs a poker hand classification study using the dataset from the UCI Machine Learning Repository (Cattral & Oppacher, 2002). The implementation uses Python and runs in Visual Studio Code. The program performs data preprocessing, feature transformation, model training, evaluation, and visualization.

---

# Program Description
The program loads the poker hand dataset, assigns column names, and separates feature measurements from class labels. It then splits the dataset into training and testing sets using a stratified 80/20 split.

The program applies SMOTE (Synthetic Minority Oversampling Technique) only to the training data to balance class distribution. This step prevents data leakage and preserves the integrity of the testing set.

The program applies feature scaling using StandardScaler. It fits the scaler on the training data and applies the transformation to both training and testing sets.

The program evaluates three feature representations:

- **Baseline (Scaled Features):** uses standardized numeric features  
- **LDA (Dimensionality Reduction):** projects scaled features into a lower-dimensional space  
- **One-Hot Encoding (Feature Expansion):** converts categorical values into binary feature vectors  

The program trains and evaluates the following models:

- Linear Support Vector Machine (Linear SVM)  
- Radial Basis Function Support Vector Machine (RBF SVM)  
- Random Forest  
- XGBoost  

The program compares model performance across all feature representations and selects the best-performing configuration for each model based on classification accuracy.

---

# Dependencies
The program requires the following Python libraries:

- numpy  
- pandas  
- matplotlib  
- scikit-learn  
- imbalanced-learn  
- xgboost  

Install all dependencies with:

pip install numpy pandas matplotlib scikit-learn imbalanced-learn xgboost  

---

# Input File Format
The program expects the dataset file:

poker-hand-training-true.data  

Place this file in the same directory as the Python script.

---

# Running the Program
1. Open the Python file: poker_hand_classification_study.py  
2. Place the dataset file in the same directory  
3. Install all required dependencies  
4. Run the program  

---

# Outputs
The program creates a folder named **results** that contains:

- Class distribution plots (before and after SMOTE)  
- Feature correlation matrix  
- LDA visualization  
- t-SNE visualization  
- Model accuracy comparison plots  
- Confusion matrix heatmaps  

The console outputs:

- Training and testing dataset sizes  
- Classification accuracy for each model and feature method  
- Best-performing configuration for each model  
- A summary table of all model accuracies  
