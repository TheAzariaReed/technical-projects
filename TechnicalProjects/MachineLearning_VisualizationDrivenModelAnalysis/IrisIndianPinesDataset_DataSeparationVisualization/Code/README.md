# Project Execution Guide

## Dependencies
Install the following Python libraries before running the project:

```bash
pip install numpy pandas matplotlib scipy scikit-learn
```

## Required Input Files
The following files must remain in the `Code/` directory:

```text
indianR.mat
indian_gth.mat
prism.py
```

## Running the Program

1. Open a terminal in the project directory.
2. Navigate to the `Code/` folder.
3. Run the program using:

```bash
python prism.py
```

## Expected Output

After execution, the program generates:

- PCA explained variance visualizations
- PCA and LDA projection plots
- Classification accuracy comparison plots
- Indian Pines class-wise accuracy table
- Saved visualization outputs inside:

```text
Code/results/
```

## Output Files
Generated outputs include:

```text
Figure_01.png through Figure_12.png
Table_1_Indian_Pines_PCA_30_Percent_Class-wise_Accuracy.csv
Table_1_Indian_Pines_PCA_30_Percent_Class-wise_Accuracy.png
```

## Notes
- The program uses the Iris and Indian Pines datasets for dimensionality reduction and classification analysis.
- Existing files inside the `results/` directory may be overwritten during execution.
