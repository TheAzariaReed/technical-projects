# Project Execution Guide

## Dependencies
Install the following Python libraries before running the project:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Required Input Files
The following file must remain in the `Code/` directory:

```text
iris.py
```

## Running the Program

1. Open a terminal in the project directory.
2. Navigate to the `Code/` folder.
3. Run the program using:

```bash
python iris.py
```

## Expected Output

After execution, the program generates:

- Correlation heatmap visualization
- Feature distribution dot plots
- Linear regression prediction outputs
- RMSE evaluation metrics

Generated visualizations include:

```text
Correlation Heatmap.png
Petal Length Dot Plot.png
Petal Width Dot Plot.png
Sepal Length Dot Plot.png
Sepal Width Dot Plot.png
```

The program also produces numerical regression outputs inside:

```text
Numeric Outputs.txt
```

## Notes
- The program uses the Iris dataset for visualization and regression analysis.
- Existing output files may be overwritten during execution.
