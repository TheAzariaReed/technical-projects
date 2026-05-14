# Poker Hand — AI Model Reliability Study

## Overview
This project evaluates the reliability and classification performance of multiple machine learning models using poker hand data. The study emphasizes **model consistency, feature representation, comparative validation, and diagnostic evaluation** across several supervised learning approaches.

## Objective
- Compare classification performance across multiple machine learning models
- Evaluate the impact of feature representation methods on reliability
- Assess class-level prediction behavior using diagnostic analysis
- Identify strengths and limitations in model generalization

## Methodology
- Data preprocessing and categorical feature handling
- Class balancing and distribution analysis
- Comparative evaluation using:
  - XGBoost
  - Support Vector Machines (Linear and RBF)
  - Random Forest
- Feature representation analysis using:
  - Raw feature encoding
  - Linear Discriminant Analysis (LDA)
  - One-hot encoding
- Performance validation using:
  - Accuracy metrics
  - Confusion matrices
  - Comparative visualization techniques

## Evidence
- **Technical Report:** `Writing/PokerHand_ReliabilityStudy_Report.pdf`
- **Code:** `Code/`
- **Briefing Deck:** `Presentations/PokerHand_ReliabilityStudy_Briefing.pdf`

## Key Findings
- Feature representation methods significantly influence classification reliability
- One-hot encoding improved performance consistency across several models
- Diagnostic confusion matrices revealed model-specific classification weaknesses
- Comparative evaluation identified measurable tradeoffs between accuracy and generalization behavior

## Limitations
- Results reflect the dataset structure and preprocessing assumptions used in this study
- Model performance may vary under alternative balancing strategies or feature engineering approaches
- Reliability conclusions remain bounded to the evaluated classification framework