import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from imblearn.over_sampling import SMOTE
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# Load the Poker Hand dataset from a CSV file and rename every column.
def LoadPokerHandDatasetFromCsvFile(csvFileName):
    pokerHandDataFrame = pd.read_csv(csvFileName, header=None)
    pokerHandDataFrame.columns = [
        "S1", "C1", "S2", "C2", "S3", "C3", "S4", "C4", "S5", "C5", "CLASS"
    ]
    return pokerHandDataFrame


# Split the full table into feature columns and class labels.
def SplitFeatureColumnsAndClassLabels(pokerHandDataFrame):
    featureMeasurementGrid = pokerHandDataFrame.drop("CLASS", axis=1)
    classLabelSeries = pokerHandDataFrame["CLASS"]
    return featureMeasurementGrid, classLabelSeries

# Build domain knowledge features.
def BuildDomainKnowledgeFeatures(featureMeasurementGrid):
    featureMeasurementGrid = featureMeasurementGrid.copy()

    rankColumns = ["C1", "C2", "C3", "C4", "C5"]
    uniqueRankCountList = []
    rankRangeList = []

    for _, cardRow in featureMeasurementGrid.iterrows():
        cardRanks = sorted(cardRow[rankColumns].tolist())

        rankRange = max(cardRanks) - min(cardRanks)
        uniqueRankCount = int(len(set(cardRanks)))

        rankRangeList.append(rankRange)
        uniqueRankCountList.append(uniqueRankCount)

    featureMeasurementGrid["Unique_Rank_Count"] = uniqueRankCountList
    featureMeasurementGrid["Rank_Range"] = rankRangeList

    return featureMeasurementGrid


# Split the data into training and testing groups before any resampling.
def BuildTrainingAndTestingSets(featureMeasurementGrid, classLabelSeries):
    return train_test_split(
        featureMeasurementGrid,
        classLabelSeries,
        test_size=0.2,
        random_state=42,
        stratify=classLabelSeries
    )


# Balance only the training class distribution with the same SMOTE settings.
def BalanceTrainingClassDistributionWithSmote(trainingFeatureMeasurements, trainingClassLabels):
    smoteResampler = SMOTE(k_neighbors=2, random_state=42)
    resampledTrainingFeatureMeasurements, resampledTrainingClassLabels = smoteResampler.fit_resample(
        trainingFeatureMeasurements,
        trainingClassLabels
    )

    resampledTrainingFeatureMeasurements = resampledTrainingFeatureMeasurements.astype(np.float32)
    return resampledTrainingFeatureMeasurements, resampledTrainingClassLabels

# Build class weights for imbalance handling.
def BuildClassWeights(trainingClassLabels):
    classCounts = trainingClassLabels.value_counts().to_dict()
    totalSamples = len(trainingClassLabels)
    classWeights = {cls: totalSamples / (len(classCounts) * count) for cls, count in classCounts.items()}
    return classWeights


# Scale the training data, then scale the testing data with the fitted scaler.
def ScaleTrainingAndTestingMeasurements(trainingFeatureMeasurements, testingFeatureMeasurements):
    featureScaler = StandardScaler(copy=False)

    scaledTrainingFeatureMeasurements = featureScaler.fit_transform(trainingFeatureMeasurements.astype(np.float32))
    scaledTestingFeatureMeasurements = featureScaler.transform(testingFeatureMeasurements.astype(np.float32))

    return scaledTrainingFeatureMeasurements, scaledTestingFeatureMeasurements, featureScaler


# Build one-hot encoded versions of the training and testing data.
def BuildOneHotEncodedTrainingAndTestingMeasurements(trainingFeatureMeasurements, testingFeatureMeasurements):
    try:
        oneHotEncoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    except TypeError:
        oneHotEncoder = OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)

    oneHotTrainingFeatureMeasurements = oneHotEncoder.fit_transform(trainingFeatureMeasurements)
    oneHotTestingFeatureMeasurements = oneHotEncoder.transform(testingFeatureMeasurements)

    return oneHotTrainingFeatureMeasurements, oneHotTestingFeatureMeasurements, oneHotEncoder


# Reduce the scaled measurements with the same LDA logic.
def ProjectTrainingAndTestingMeasurementsIntoLdaSpace(
    scaledTrainingFeatureMeasurements,
    scaledTestingFeatureMeasurements,
    trainingClassLabels
):
    ldaComponentCount = min(len(set(trainingClassLabels)) - 1, scaledTrainingFeatureMeasurements.shape[1])
    ldaProjector = LDA(n_components=ldaComponentCount)

    ldaTrainingFeatureMeasurements = ldaProjector.fit_transform(
        scaledTrainingFeatureMeasurements,
        trainingClassLabels
    ).astype(np.float32)

    ldaScaler = StandardScaler()
    ldaTrainingFeatureMeasurements = ldaScaler.fit_transform(ldaTrainingFeatureMeasurements)

    ldaTestingFeatureMeasurements = ldaProjector.transform(scaledTestingFeatureMeasurements).astype(np.float32)
    ldaTestingFeatureMeasurements = ldaScaler.transform(ldaTestingFeatureMeasurements)

    return ldaTrainingFeatureMeasurements, ldaTestingFeatureMeasurements, ldaProjector


# Create a clean results folder for every saved figure.
def BuildResultsFolder():
    resultsFolder = Path("results")
    resultsFolder.mkdir(exist_ok=True)
    return resultsFolder


# Save a finished figure with a consistent figure number and caption.
def SaveFinishedFigure(figurePaper, resultsFolder, figureNumber, figureTitle, figureCaption):
    cleanedFigureCaption = " ".join(figureCaption.strip().split()).rstrip(".")
    captionText = f"Figure {int(figureNumber)}. {cleanedFigureCaption}."
    figurePaper.subplots_adjust(bottom=0.18)
    figurePaper.text(0.5, 0.02, captionText, ha="center", va="bottom", fontsize=10)

    safeFigureStem = f"Figure_{figureNumber:02d}_{figureTitle.replace(' ', '_').replace('/', '_')}"
    figurePaper.savefig(resultsFolder / f"{safeFigureStem}.png", dpi=300, bbox_inches="tight")


# Save a finished table image with a consistent table number and caption.
def SaveFinishedTable(tablePaper, resultsFolder, tableNumber, tableTitle, tableCaption):
    cleanedTableCaption = " ".join(tableCaption.strip().split()).rstrip(".")
    captionText = f"Table {int(tableNumber)}. {cleanedTableCaption}."
    tablePaper.subplots_adjust(bottom=0.2)
    tablePaper.text(0.5, 0.02, captionText, ha="center", va="bottom", fontsize=10)

    safeTableStem = f"Table_{tableNumber:02d}_{tableTitle.replace(' ', '_').replace('/', '_')}"
    tablePaper.savefig(resultsFolder / f"{safeTableStem}.png", dpi=300, bbox_inches="tight")


# Plot the original class distribution before SMOTE.
def PlotOriginalClassDistribution(classLabelSeries, resultsFolder, figureNumber):
    figurePaper, drawingArea = plt.subplots(figsize=(10, 5))

    classCounts = classLabelSeries.value_counts().sort_index()
    drawingArea.bar(classCounts.index.astype(str), classCounts.values)

    drawingArea.set_title("Original Poker Hand Class Distribution")
    drawingArea.set_xlabel("Class Label")
    drawingArea.set_ylabel("Count")
    drawingArea.grid(True, axis="y")

    figurePaper.tight_layout()
    SaveFinishedFigure(
        figurePaper,
        resultsFolder,
        figureNumber,
        "Original_Class_Distribution",
        "Original Poker Hand class distribution"
    )
    plt.show()
    plt.close(figurePaper)


# Plot the balanced training class distribution after SMOTE.
def PlotBalancedTrainingClassDistribution(resampledTrainingClassLabels, resultsFolder, figureNumber):
    figurePaper, drawingArea = plt.subplots(figsize=(10, 5))

    classCounts = pd.Series(resampledTrainingClassLabels).value_counts().sort_index()
    drawingArea.bar(classCounts.index.astype(str), classCounts.values)

    drawingArea.set_title("Balanced Training Class Distribution After SMOTE")
    drawingArea.set_xlabel("Class Label")
    drawingArea.set_ylabel("Count")
    drawingArea.grid(True, axis="y")

    figurePaper.tight_layout()
    SaveFinishedFigure(
        figurePaper,
        resultsFolder,
        figureNumber,
        "Balanced_Training_Class_Distribution",
        "Balanced Poker Hand training class distribution after SMOTE"
    )
    plt.show()
    plt.close(figurePaper)


# Plot a correlation matrix for the original feature columns.
def PlotFeatureCorrelationMatrix(featureMeasurementGrid, resultsFolder, figureNumber):
    correlationMatrix = featureMeasurementGrid.corr()

    figurePaper, drawingArea = plt.subplots(figsize=(10, 8))
    heatMapImage = drawingArea.imshow(correlationMatrix, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)

    drawingArea.set_title("Poker Hand Feature Correlation Matrix")
    drawingArea.set_xticks(range(len(correlationMatrix.columns)))
    drawingArea.set_yticks(range(len(correlationMatrix.columns)))
    drawingArea.set_xticklabels(correlationMatrix.columns, rotation=45, ha="right")
    drawingArea.set_yticklabels(correlationMatrix.columns)

    colorBar = figurePaper.colorbar(heatMapImage, ax=drawingArea)
    colorBar.set_label("Correlation")

    figurePaper.tight_layout()
    SaveFinishedFigure(
        figurePaper,
        resultsFolder,
        figureNumber,
        "Feature_Correlation_Matrix",
        "Poker Hand feature correlation matrix"
    )
    plt.show()
    plt.close(figurePaper)


# Plot the first two LDA directions to show class separation after reduction.
def PlotFirstTwoLdaDirections(ldaTrainingFeatureMeasurements, trainingClassLabels, resultsFolder, figureNumber):
    if ldaTrainingFeatureMeasurements.shape[1] < 2:
        print("LDA produced fewer than 2 components, so the first-two-directions plot was skipped.")
        return

    sampleSize = 5000
    randomNumberGenerator = np.random.default_rng(42)
    selectedSampleSize = min(sampleSize, len(ldaTrainingFeatureMeasurements))
    selectedRowIndexes = randomNumberGenerator.choice(
        len(ldaTrainingFeatureMeasurements),
        size=selectedSampleSize,
        replace=False
    )

    xCoordinates = ldaTrainingFeatureMeasurements[selectedRowIndexes, 0]
    yCoordinates = ldaTrainingFeatureMeasurements[selectedRowIndexes, 1]
    classLabels = np.array(trainingClassLabels)[selectedRowIndexes]

    xCoordinates = xCoordinates + randomNumberGenerator.normal(0, 0.02, size=xCoordinates.shape)
    yCoordinates = yCoordinates + randomNumberGenerator.normal(0, 0.02, size=yCoordinates.shape)

    figurePaper, drawingArea = plt.subplots(figsize=(9, 6))

    uniqueClassNumbers = np.unique(classLabels)
    colorPicture = drawingArea.scatter(
        xCoordinates,
        yCoordinates,
        c=classLabels,
        cmap="tab10",
        s=10,
        alpha=0.4
    )

    drawingArea.set_title("Poker Hand LDA First Two Directions")
    drawingArea.set_xlabel("LD1")
    drawingArea.set_ylabel("LD2")
    drawingArea.grid(True)

    colorBar = figurePaper.colorbar(colorPicture, ax=drawingArea)
    colorBar.set_label("Class Label")
    colorBar.set_ticks(uniqueClassNumbers)

    figurePaper.tight_layout()
    SaveFinishedFigure(
        figurePaper,
        resultsFolder,
        figureNumber,
        "LDA_First_Two_Directions",
        "First two LDA directions for Poker Hand training samples"
    )
    plt.show()
    plt.close(figurePaper)


# Plot the first two t-SNE directions from one-hot features for visualization only.
def PlotFirstTwoTsneDirectionsFromOneHot(oneHotTrainingFeatureMeasurements, trainingClassLabels, resultsFolder, figureNumber):
    denseOneHotTrainingFeatureMeasurements = oneHotTrainingFeatureMeasurements.toarray()

    maximumVisualizationSampleCount = 6000
    if denseOneHotTrainingFeatureMeasurements.shape[0] > maximumVisualizationSampleCount:
        randomNumberGenerator = np.random.default_rng(42)
        selectedRowIndexes = randomNumberGenerator.choice(
            denseOneHotTrainingFeatureMeasurements.shape[0],
            size=maximumVisualizationSampleCount,
            replace=False
        )
        denseOneHotTrainingFeatureMeasurements = denseOneHotTrainingFeatureMeasurements[selectedRowIndexes]
        trainingClassLabels = np.array(trainingClassLabels)[selectedRowIndexes]

    tsneProjector = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42
    )

    tsneTrainingFeatureMeasurements = tsneProjector.fit_transform(denseOneHotTrainingFeatureMeasurements).astype(np.float32)

    figurePaper, drawingArea = plt.subplots(figsize=(9, 6))

    uniqueClassNumbers = np.unique(trainingClassLabels)
    colorPicture = drawingArea.scatter(
        tsneTrainingFeatureMeasurements[:, 0],
        tsneTrainingFeatureMeasurements[:, 1],
        c=trainingClassLabels,
        cmap="tab10",
        s=18,
        alpha=0.75
    )

    drawingArea.set_title("Poker Hand One-Hot t-SNE First Two Directions")
    drawingArea.set_xlabel("t-SNE 1")
    drawingArea.set_ylabel("t-SNE 2")
    drawingArea.grid(True)

    colorBar = figurePaper.colorbar(colorPicture, ax=drawingArea)
    colorBar.set_label("Class Label")
    colorBar.set_ticks(uniqueClassNumbers)

    figurePaper.tight_layout()
    SaveFinishedFigure(
        figurePaper,
        resultsFolder,
        figureNumber,
        "One_Hot_TSNE_First_Two_Directions",
        "First two t-SNE directions from one-hot Poker Hand training samples"
    )
    plt.show()
    plt.close(figurePaper)


# Train an XGBoost classifier on the given measurements and return predictions.
def RunXgboostClassification(trainingFeatureMeasurements, testingFeatureMeasurements, trainingClassLabels):
    classWeights = BuildClassWeights(trainingClassLabels)
    sampleWeights = np.array([classWeights[label] for label in trainingClassLabels])

    xgboostClassifier = XGBClassifier(
        objective="multi:softmax",
        num_class=len(np.unique(trainingClassLabels)),
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_lambda=1.0,
        reg_alpha=0.1,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        eval_metric="mlogloss"
    )

    xgboostClassifier.fit(trainingFeatureMeasurements, trainingClassLabels, sample_weight=sampleWeights)
    predictedTestingLabels = xgboostClassifier.predict(testingFeatureMeasurements)
    return xgboostClassifier, predictedTestingLabels


# Train a linear SVM on the given measurements and return predictions.
def RunLinearSvmClassification(trainingFeatureMeasurements, testingFeatureMeasurements, trainingClassLabels):
    linearSupportVectorMachine = SVC(kernel="linear", cache_size=1000)
    linearSupportVectorMachine.fit(trainingFeatureMeasurements, trainingClassLabels)
    predictedTestingLabels = linearSupportVectorMachine.predict(testingFeatureMeasurements)
    return linearSupportVectorMachine, predictedTestingLabels


# Train an RBF SVM on the given measurements and return predictions.
def RunRbfSvmClassification(trainingFeatureMeasurements, testingFeatureMeasurements, trainingClassLabels):
    radialBasisSupportVectorMachine = SVC(kernel="rbf", cache_size=1000)
    radialBasisSupportVectorMachine.fit(trainingFeatureMeasurements, trainingClassLabels)
    predictedTestingLabels = radialBasisSupportVectorMachine.predict(testingFeatureMeasurements)
    return radialBasisSupportVectorMachine, predictedTestingLabels

# Train a Random Forest classifier on the given measurements and return predictions.
def RunRandomForestClassification(trainingFeatureMeasurements, testingFeatureMeasurements, trainingClassLabels):
    randomForestClassifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42
    )

    randomForestClassifier.fit(trainingFeatureMeasurements, trainingClassLabels)
    predictedTestingLabels = randomForestClassifier.predict(testingFeatureMeasurements)

    return randomForestClassifier, predictedTestingLabels

# Measure plain classification accuracy for a predicted label set.
def MeasureClassificationAccuracy(testingClassLabels, predictedTestingLabels):
    return accuracy_score(testingClassLabels, predictedTestingLabels)


# Draw and save a confusion matrix heat map.
def PlotConfusionMatrixHeatMap(
    testingClassLabels,
    predictedTestingLabels,
    chartTitle,
    resultsFolder,
    figureNumber,
    figureCaption
):
    classNumbersInOrder = np.unique(np.concatenate([np.array(testingClassLabels), np.array(predictedTestingLabels)]))
    confusionMatrixGrid = confusion_matrix(
        testingClassLabels,
        predictedTestingLabels,
        labels=classNumbersInOrder
    )

    figurePaper, drawingArea = plt.subplots(figsize=(8, 6))
    heatMapImage = drawingArea.imshow(confusionMatrixGrid, cmap="Blues", aspect="auto")

    drawingArea.set_title(chartTitle)
    drawingArea.set_xlabel("Predicted Class")
    drawingArea.set_ylabel("True Class")
    drawingArea.set_xticks(range(len(classNumbersInOrder)))
    drawingArea.set_yticks(range(len(classNumbersInOrder)))
    drawingArea.set_xticklabels(classNumbersInOrder)
    drawingArea.set_yticklabels(classNumbersInOrder)

    for rowIndex in range(confusionMatrixGrid.shape[0]):
        for columnIndex in range(confusionMatrixGrid.shape[1]):
            drawingArea.text(
                columnIndex,
                rowIndex,
                confusionMatrixGrid[rowIndex, columnIndex],
                ha="center",
                va="center",
                color="black"
            )

    colorBar = figurePaper.colorbar(heatMapImage, ax=drawingArea)
    colorBar.set_label("Count")

    figurePaper.tight_layout()
    SaveFinishedFigure(figurePaper, resultsFolder, figureNumber, chartTitle, figureCaption)
    plt.show()
    plt.close(figurePaper)


# Compare model accuracies in one clean plot.
def PlotModelAccuracyComparison(modelAccuracyBook, chartTitle, resultsFolder, figureNumber, figureFileName, figureCaption):
    figurePaper, drawingArea = plt.subplots(figsize=(8, 5))

    modelNames = list(modelAccuracyBook.keys())
    modelAccuracies = list(modelAccuracyBook.values())

    drawingArea.bar(modelNames, modelAccuracies)
    drawingArea.set_title(chartTitle)
    drawingArea.set_xlabel("Model")
    drawingArea.set_ylabel("Classification Accuracy")
    drawingArea.set_ylim(0, 1)
    drawingArea.grid(True, axis="y")

    for modelIndex, modelAccuracy in enumerate(modelAccuracies):
        drawingArea.text(
            modelIndex,
            modelAccuracy + 0.01,
            f"{modelAccuracy:.4f}",
            ha="center"
        )

    figurePaper.tight_layout()
    SaveFinishedFigure(figurePaper, resultsFolder, figureNumber, figureFileName, figureCaption)
    plt.show()
    plt.close(figurePaper)


# Save a compact results table.
def SaveModelAccuracyTable(modelAccuracyBook, resultsFolder):
    accuracySummaryTable = pd.DataFrame(
        {
            "Model": list(modelAccuracyBook.keys()),
            "Classification Accuracy": list(modelAccuracyBook.values())
        }
    )

    accuracySummaryTable.to_csv(resultsFolder / "Table_1_Model_Accuracy_Summary_Raw_LDA_And_One_Hot.csv", index=False)
    return accuracySummaryTable


# Plot and save the model accuracy summary as a captioned table image.
def PlotModelAccuracySummaryTable(accuracySummaryTable, resultsFolder, tableNumber, tableTitle, tableCaption):
    tablePaper, drawingArea = plt.subplots(figsize=(10, 5))
    drawingArea.axis("off")

    displayTable = accuracySummaryTable.copy()
    displayTable["Classification Accuracy"] = displayTable["Classification Accuracy"].map(lambda value: f"{value:.4f}")

    renderedTable = drawingArea.table(
        cellText=displayTable.values,
        colLabels=displayTable.columns,
        cellLoc="center",
        loc="center"
    )
    renderedTable.auto_set_font_size(False)
    renderedTable.set_fontsize(10)
    renderedTable.scale(1, 1.25)

    tablePaper.tight_layout()
    SaveFinishedTable(tablePaper, resultsFolder, tableNumber, tableTitle, tableCaption)
    plt.show()
    plt.close(tablePaper)


# Pick the best result for a single model across raw, LDA, and one-hot inputs.
def ChooseBestModelVersion(
    rawAccuracy,
    rawPredictedTestingLabels,
    ldaAccuracy,
    ldaPredictedTestingLabels,
    oneHotAccuracy,
    oneHotPredictedTestingLabels
):
    if oneHotAccuracy >= ldaAccuracy and oneHotAccuracy >= rawAccuracy:
        return oneHotAccuracy, oneHotPredictedTestingLabels, "With One-Hot"
    if ldaAccuracy >= rawAccuracy:
        return ldaAccuracy, ldaPredictedTestingLabels, "With LDA"
    return rawAccuracy, rawPredictedTestingLabels, "Baseline"


# Run the full Poker Hand study from start to finish.
def RunPokerHandClassificationStudy():
    resultsFolder = BuildResultsFolder()

    pokerHandDataFrame = LoadPokerHandDatasetFromCsvFile("poker-hand-training-true.data")
    originalFeatureMeasurementGrid, originalClassLabelSeries = SplitFeatureColumnsAndClassLabels(pokerHandDataFrame)
    originalFeatureMeasurementGrid = BuildDomainKnowledgeFeatures(originalFeatureMeasurementGrid)

    PlotOriginalClassDistribution(originalClassLabelSeries, resultsFolder, 1)
    PlotFeatureCorrelationMatrix(originalFeatureMeasurementGrid, resultsFolder, 2)

    (
        trainingFeatureMeasurements,
        testingFeatureMeasurements,
        trainingClassLabels,
        testingClassLabels
    ) = BuildTrainingAndTestingSets(
        originalFeatureMeasurementGrid,
        originalClassLabelSeries
    )

    (
        resampledTrainingFeatureMeasurements,
        resampledTrainingClassLabels
    ) = BalanceTrainingClassDistributionWithSmote(
        trainingFeatureMeasurements,
        trainingClassLabels
    )

    PlotBalancedTrainingClassDistribution(resampledTrainingClassLabels, resultsFolder, 3)

    print("Training set:", resampledTrainingFeatureMeasurements.shape)
    print("Testing set:", testingFeatureMeasurements.shape)

    (
        scaledTrainingFeatureMeasurements,
        scaledTestingFeatureMeasurements,
        featureScaler
    ) = ScaleTrainingAndTestingMeasurements(
        resampledTrainingFeatureMeasurements,
        testingFeatureMeasurements
    )

    (
        oneHotTrainingFeatureMeasurements,
        oneHotTestingFeatureMeasurements,
        oneHotEncoder
    ) = BuildOneHotEncodedTrainingAndTestingMeasurements(
        resampledTrainingFeatureMeasurements,
        testingFeatureMeasurements
    )

    (
        ldaTrainingFeatureMeasurements,
        ldaTestingFeatureMeasurements,
        ldaProjector
    ) = ProjectTrainingAndTestingMeasurementsIntoLdaSpace(
        scaledTrainingFeatureMeasurements,
        scaledTestingFeatureMeasurements,
        resampledTrainingClassLabels
    )

    print("Resampled training:", resampledTrainingFeatureMeasurements.shape)
    print("LDA training:", ldaTrainingFeatureMeasurements.shape)
    print("One-Hot training:", oneHotTrainingFeatureMeasurements.shape)
    print("Testing set:", testingFeatureMeasurements.shape)
    print("LDA testing:", ldaTestingFeatureMeasurements.shape)
    print("One-Hot testing:", oneHotTestingFeatureMeasurements.shape)

    PlotFirstTwoLdaDirections(ldaTrainingFeatureMeasurements, resampledTrainingClassLabels, resultsFolder, 4)
    PlotFirstTwoTsneDirectionsFromOneHot(oneHotTrainingFeatureMeasurements, resampledTrainingClassLabels, resultsFolder, 5)

    (
        rawXgboostClassifier,
        rawXgboostPredictedTestingLabels
    ) = RunXgboostClassification(
        scaledTrainingFeatureMeasurements,
        scaledTestingFeatureMeasurements,
        resampledTrainingClassLabels
    )

    (
        rawLinearSupportVectorMachine,
        rawLinearPredictedTestingLabels
    ) = RunLinearSvmClassification(
        scaledTrainingFeatureMeasurements,
        scaledTestingFeatureMeasurements,
        resampledTrainingClassLabels
    )

    (
        rawRadialBasisSupportVectorMachine,
        rawRadialBasisPredictedTestingLabels
    ) = RunRbfSvmClassification(
        scaledTrainingFeatureMeasurements,
        scaledTestingFeatureMeasurements,
        resampledTrainingClassLabels
    )
    (
        rawRandomForestClassifier,
        rawRandomForestPredictedTestingLabels
    ) = RunRandomForestClassification(
        scaledTrainingFeatureMeasurements,
        scaledTestingFeatureMeasurements,
        resampledTrainingClassLabels
    )

    rawXgboostAccuracy = MeasureClassificationAccuracy(
        testingClassLabels,
        rawXgboostPredictedTestingLabels
    )

    rawRadialBasisSvmAccuracy = MeasureClassificationAccuracy(
        testingClassLabels,
        rawRadialBasisPredictedTestingLabels
    )

    rawLinearSvmAccuracy = MeasureClassificationAccuracy(
        testingClassLabels,
        rawLinearPredictedTestingLabels
    )

    rawRandomForestAccuracy = MeasureClassificationAccuracy(
        testingClassLabels,
        rawRandomForestPredictedTestingLabels
    )

    rawModelAccuracyBook = {
        "Linear SVM": rawLinearSvmAccuracy,
        "RBF SVM": rawRadialBasisSvmAccuracy,
        "XGBoost": rawXgboostAccuracy,
        "Random Forest": rawRandomForestAccuracy
    }

    PlotModelAccuracyComparison(
        rawModelAccuracyBook,
        "Model Accuracy Comparison",
        resultsFolder,
        6,
        "Model_Accuracy_Comparison",
        "Model accuracy comparison on baseline scaled features"
    )

    (
        ldaXgboostClassifier,
        ldaXgboostPredictedTestingLabels
    ) = RunXgboostClassification(
        ldaTrainingFeatureMeasurements,
        ldaTestingFeatureMeasurements,
        resampledTrainingClassLabels
    )

    (
        ldaRandomForestClassifier,
        ldaRandomForestPredictedTestingLabels
    ) = RunRandomForestClassification(
        ldaTrainingFeatureMeasurements,
        ldaTestingFeatureMeasurements,
        resampledTrainingClassLabels
    )

    (
        ldaLinearSupportVectorMachine,
        ldaLinearPredictedTestingLabels
    ) = RunLinearSvmClassification(
        ldaTrainingFeatureMeasurements,
        ldaTestingFeatureMeasurements,
        resampledTrainingClassLabels
    )

    (
        ldaRadialBasisSupportVectorMachine,
        ldaRadialBasisPredictedTestingLabels
    ) = RunRbfSvmClassification(
        ldaTrainingFeatureMeasurements,
        ldaTestingFeatureMeasurements,
        resampledTrainingClassLabels
    )

    ldaXgboostAccuracy = MeasureClassificationAccuracy(
        testingClassLabels,
        ldaXgboostPredictedTestingLabels
    )

    ldaRadialBasisSvmAccuracy = MeasureClassificationAccuracy(
        testingClassLabels,
        ldaRadialBasisPredictedTestingLabels
    )

    ldaLinearSvmAccuracy = MeasureClassificationAccuracy(
        testingClassLabels,
        ldaLinearPredictedTestingLabels
    )

    ldaRandomForestAccuracy = MeasureClassificationAccuracy(
        testingClassLabels,
        ldaRandomForestPredictedTestingLabels
    )

    ldaModelAccuracyBook = {
        "Linear SVM": ldaLinearSvmAccuracy,
        "RBF SVM": ldaRadialBasisSvmAccuracy,
        "XGBoost": ldaXgboostAccuracy,
        "Random Forest": ldaRandomForestAccuracy
    }

    PlotModelAccuracyComparison(
        ldaModelAccuracyBook,
        "Model Accuracy Comparison With LDA",
        resultsFolder,
        7,
        "Model_Accuracy_Comparison_With_LDA",
        "Model accuracy comparison with LDA features"
    )

    (
        oneHotXgboostClassifier,
        oneHotXgboostPredictedTestingLabels
    ) = RunXgboostClassification(
        oneHotTrainingFeatureMeasurements,
        oneHotTestingFeatureMeasurements,
        resampledTrainingClassLabels
    )

    (
        oneHotLinearSupportVectorMachine,
        oneHotLinearPredictedTestingLabels
    ) = RunLinearSvmClassification(
        oneHotTrainingFeatureMeasurements,
        oneHotTestingFeatureMeasurements,
        resampledTrainingClassLabels
    )

    (
        oneHotRadialBasisSupportVectorMachine,
        oneHotRadialBasisPredictedTestingLabels
    ) = RunRbfSvmClassification(
        oneHotTrainingFeatureMeasurements,
        oneHotTestingFeatureMeasurements,
        resampledTrainingClassLabels
    )

    (
        oneHotRandomForestClassifier,
        oneHotRandomForestPredictedTestingLabels
    ) = RunRandomForestClassification(
        oneHotTrainingFeatureMeasurements,
        oneHotTestingFeatureMeasurements,
        resampledTrainingClassLabels
    )

    oneHotXgboostAccuracy = MeasureClassificationAccuracy(
        testingClassLabels,
        oneHotXgboostPredictedTestingLabels
    )

    oneHotRadialBasisSvmAccuracy = MeasureClassificationAccuracy(
        testingClassLabels,
        oneHotRadialBasisPredictedTestingLabels
    )

    oneHotLinearSvmAccuracy = MeasureClassificationAccuracy(
        testingClassLabels,
        oneHotLinearPredictedTestingLabels
    )

    oneHotRandomForestAccuracy = MeasureClassificationAccuracy(
        testingClassLabels,
        oneHotRandomForestPredictedTestingLabels
    )

    oneHotModelAccuracyBook = {
        "Linear SVM": oneHotLinearSvmAccuracy,
        "RBF SVM": oneHotRadialBasisSvmAccuracy,
        "XGBoost": oneHotXgboostAccuracy,
        "Random Forest": oneHotRandomForestAccuracy
    }

    PlotModelAccuracyComparison(
        oneHotModelAccuracyBook,
        "Model Accuracy Comparison With One-Hot",
        resultsFolder,
        8,
        "Model_Accuracy_Comparison_With_One_Hot",
        "Model accuracy comparison with one-hot encoded features"
    )

    bestXgboostAccuracy, bestXgboostPredictedTestingLabels, bestXgboostVersion = ChooseBestModelVersion(
        rawXgboostAccuracy,
        rawXgboostPredictedTestingLabels,
        ldaXgboostAccuracy,
        ldaXgboostPredictedTestingLabels,
        oneHotXgboostAccuracy,
        oneHotXgboostPredictedTestingLabels
    )

    bestRadialBasisSvmAccuracy, bestRadialBasisPredictedTestingLabels, bestRadialBasisSvmVersion = ChooseBestModelVersion(
        rawRadialBasisSvmAccuracy,
        rawRadialBasisPredictedTestingLabels,
        ldaRadialBasisSvmAccuracy,
        ldaRadialBasisPredictedTestingLabels,
        oneHotRadialBasisSvmAccuracy,
        oneHotRadialBasisPredictedTestingLabels
    )

    bestLinearSvmAccuracy, bestLinearPredictedTestingLabels, bestLinearSvmVersion = ChooseBestModelVersion(
        rawLinearSvmAccuracy,
        rawLinearPredictedTestingLabels,
        ldaLinearSvmAccuracy,
        ldaLinearPredictedTestingLabels,
        oneHotLinearSvmAccuracy,
        oneHotLinearPredictedTestingLabels
    )

    bestRandomForestAccuracy, bestRandomForestPredictedTestingLabels, bestRandomForestVersion = ChooseBestModelVersion(
        rawRandomForestAccuracy,
        rawRandomForestPredictedTestingLabels,
        ldaRandomForestAccuracy,
        ldaRandomForestPredictedTestingLabels,
        oneHotRandomForestAccuracy,
        oneHotRandomForestPredictedTestingLabels
    )

    print(f"Linear SVM Accuracy: {rawLinearSvmAccuracy:.4f}")
    print(f"RBF SVM Accuracy: {rawRadialBasisSvmAccuracy:.4f}")
    print(f"XGBoost Accuracy: {rawXgboostAccuracy:.4f}")
    print(f"Random Forest Accuracy: {rawRandomForestAccuracy:.4f}")

    print(f"Linear SVM Accuracy With LDA: {ldaLinearSvmAccuracy:.4f}")
    print(f"RBF SVM Accuracy With LDA: {ldaRadialBasisSvmAccuracy:.4f}")
    print(f"XGBoost Accuracy With LDA: {ldaXgboostAccuracy:.4f}")
    print(f"Random Forest Accuracy With LDA: {ldaRandomForestAccuracy:.4f}")
    
    print(f"Linear SVM Accuracy With One-Hot: {oneHotLinearSvmAccuracy:.4f}")
    print(f"RBF SVM Accuracy With One-Hot: {oneHotRadialBasisSvmAccuracy:.4f}")
    print(f"XGBoost Accuracy With One-Hot: {oneHotXgboostAccuracy:.4f}")
    print(f"Random Forest Accuracy With One-Hot: {oneHotRandomForestAccuracy:.4f}")

    print(f"Best XGBoost Version: {bestXgboostVersion}")
    print(f"Best RBF SVM Version: {bestRadialBasisSvmVersion}")
    print(f"Best Linear SVM Version: {bestLinearSvmVersion}")
    print(f"Best Random Forest Version: {bestRandomForestVersion}")


    PlotConfusionMatrixHeatMap(
        testingClassLabels,
        bestXgboostPredictedTestingLabels,
        f"XGBoost Confusion Matrix ({bestXgboostVersion})",
        resultsFolder,
        9,
        f"XGBoost confusion matrix using the best version ({bestXgboostVersion})"
    )

    PlotConfusionMatrixHeatMap(
        testingClassLabels,
        bestRadialBasisPredictedTestingLabels,
        f"RBF SVM Confusion Matrix ({bestRadialBasisSvmVersion})",
        resultsFolder,
        10,
        f"RBF SVM confusion matrix using the best version ({bestRadialBasisSvmVersion})"
    )

    PlotConfusionMatrixHeatMap(
        testingClassLabels,
        bestLinearPredictedTestingLabels,
        f"Linear SVM Confusion Matrix ({bestLinearSvmVersion})",
        resultsFolder,
        11,
        f"Linear SVM confusion matrix using the best version ({bestLinearSvmVersion})"
    )

    PlotConfusionMatrixHeatMap(
        testingClassLabels,
        bestRandomForestPredictedTestingLabels,
        f"Random Forest Confusion Matrix ({bestRandomForestVersion})",
        resultsFolder,
        12,
        f"Random Forest confusion matrix using the best version ({bestRandomForestVersion})"
    )

    modelAccuracyBook = {
        "Linear SVM": rawLinearSvmAccuracy,
        "RBF SVM": rawRadialBasisSvmAccuracy,
        "XGBoost": rawXgboostAccuracy,
        "Random Forest": rawRandomForestAccuracy,
        "Linear SVM With LDA": ldaLinearSvmAccuracy,
        "RBF SVM With LDA": ldaRadialBasisSvmAccuracy,
        "XGBoost With LDA": ldaXgboostAccuracy,
        "Random Forest With LDA": ldaRandomForestAccuracy,
        "Linear SVM With One-Hot": oneHotLinearSvmAccuracy,
        "RBF SVM With One-Hot": oneHotRadialBasisSvmAccuracy,
        "XGBoost With One-Hot": oneHotXgboostAccuracy,
        "Random Forest With One-Hot": oneHotRandomForestAccuracy

    }

    accuracySummaryTable = SaveModelAccuracyTable(modelAccuracyBook, resultsFolder)
    PlotModelAccuracySummaryTable(
        accuracySummaryTable,
        resultsFolder,
        1,
        "Model_Accuracy_Summary_Raw_LDA_And_One_Hot",
        "Model accuracy summary across baseline, LDA, and one-hot feature settings"
    )

    print("\nAccuracy Summary")
    print(accuracySummaryTable.to_string(index=False))
    print(f"\nSaved outputs to: {resultsFolder.resolve()}")


if __name__ == "__main__":
    RunPokerHandClassificationStudy()
