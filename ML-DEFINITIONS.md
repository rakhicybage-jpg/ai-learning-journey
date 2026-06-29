# ML Definitions & Interview Prep

## Core Concepts

**Overfitting**
- Definition: Model memorizes training data instead of learning patterns
- Detection: Large gap between training accuracy and testing accuracy
- Example: 98% train, 75% test = overfitting
- Fix: Use simpler model, more data, or Random Forest

**Underfitting**
- Definition: Model is too simple, performs poorly on both train and test
- Detection: Both training and testing accuracy are low, tiny gap
- Example: 78% train, 78% test = underfitting
- Fix: Use more complex model or add more features

**Cross Validation**
- Definition: Train and test model 5 times on different data, average the scores
- Why: Single split can be lucky/unlucky — CV gives honest accuracy
- Example: 5 scores averaged = 81.38% ± 2.15%

**Feature Engineering**
- Definition: Creating new columns from existing columns to improve accuracy
- Example: FamilySize = SibSp + Parch + 1
- Result: Accuracy improved from 81.01% to 81.56%

**MAE (Mean Absolute Error)**
- Definition: Average difference between actual and predicted values
- Use: Regression problems (predicting numbers)
- Example: MAE = $3,333 means predictions are $3,333 off on average

**Confusion Matrix**
- Definition: Table showing where model is right and wrong
- Better than: Accuracy alone — shows WHICH mistakes model makes
- Key terms: False Positive, False Negative

**False Negative**
- Definition: Model predicted NO but answer was YES
- Example: Model said passenger died but they actually survived
- Real world: Cancer detector missed a cancer patient — dangerous!

## Models

**LinearRegression**
- Use when: Predicting a number (house price, temperature)
- Output: Exact number
- Weakness: Breaks at extreme values (can predict negative prices!)

**RandomForestClassifier**
- Use when: Predicting yes/no (survived/died, spam/not spam)
- How: 100 trees vote together
- Strength: Handles extremes well, less overfitting

**RandomForestRegressor**
- Use when: Predicting a number but need better accuracy than LinearRegression
- How: 100 trees average their number predictions

**DecisionTreeClassifier**
- Use when: Simple yes/no prediction, need to understand decisions
- How: One tree asking yes/no questions
- Weakness: Overfits easily

**LogisticRegression**
- Use when: Yes/no prediction, need fast and consistent model
- How: Draws a straight line between two groups
- Strength: Very consistent (low std dev)

**KNeighborsClassifier**
- Use when: Data has clear neighbourhood patterns
- How: Looks at nearest neighbours, takes majority vote
- Weakness: Sensitive to feature scaling

**SVC (Support Vector Classifier)**
- Use when: Need best possible dividing line between groups
- How: Finds maximum gap between two groups
- Weakness: Sensitive to feature scaling

**max_depth**
- max_depth=1 → 1 question  → too simple
- max_depth=5 → 5 questions → sweet spot
- max_depth=None → unlimited questions → memorizes everything

## Key Rules

1. ML models CRASH on empty values (NaN) — always fill or drop
2. ML models only understand numbers — convert text to numbers
3. Always split data 80/20 — never test on training data
4. Always compare multiple models — never assume one always wins
5. Use Cross Validation for honest comparison
6. Check feature importance after training

**Feature Scaling**
- Definition: Converting all features to same range so no feature dominates
- Why: Distance-based models measure distances — big numbers dominate unfairly
- Example: Fare=512 dominated Age=80 → after scaling both = 0 to 1
- Models that need scaling: KNeighbors, SVC
- Models that don't need scaling: Random Forest, Decision Tree

**MinMaxScaler**
- Definition: Converts all values to range 0-1
- fit_transform on training data → transform only on test data
- Result: KNeighbors 69% → 81%, SVC 67% → 81%
7. Large gap between train/test = overfitting
8. Both scores low = underfitting

**Pipeline**
- Definition: Chains multiple steps (scaling + model) into one object
- Why: Can never forget preprocessing steps
- Example: scaler → KNeighbors in one pipeline
- Result: SVC jumped from 67% to 82% with pipeline!

**StandardScaler**
- Definition: Centers data around 0 using average
- Better than MinMaxScaler for most ML models
- Above average → positive, Below average → negative
  
