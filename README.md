# Smart Credit Risk Scoring

A credit risk scoring system that predicts whether a LendingClub loan will default, to help lenders make smarter, data-driven lending decisions.

This is a personal project I built to learn credit risk modelling end to end: cleaning a large real-world dataset, exploring what drives default, comparing a simple model with a more complex one, and explaining the predictions.

## Key questions

1. Which borrower characteristics are associated with higher default risk?
2. Which features are the strongest predictors of default?
3. How well can a model rank risky borrowers above safe ones?

## Data

- **Source:** LendingClub accepted loans, 2007–2018 ([Kaggle: Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club))
- **Size:** 2.26 million loans and 151 columns before cleaning
- After cleaning: **1.37 million** finished loans (fully paid or defaulted), reduced to **16 predictors**

The data is not included in this repo because of its size. Download `accepted_2007_to_2018Q4.csv` from the link above into a `data/` folder to re-run the notebooks.

## What I did

1. **Cleaning** (`notebooks/01_EDA_lendingclub.ipynb`)
   - Turned loan status into a yes/no "default" target
   - Dropped columns with more than 40% missing values, and columns that would leak the answer (for example, payments received after the loan was issued)
   - Filled in remaining missing values and created new features such as credit age and issue year
2. **Exploratory analysis**
   - 60-month loans default about **twice as often** as 36-month loans
   - Default rates rise sharply for lower credit sub-grades (D to G)
   - Incomes under $40k and debt-to-income ratios above 30 are strong warning signs
3. **Feature selection:** removed highly correlated features, then used XGBoost feature importance to keep 16 predictors
4. **Modelling** (`notebooks/02_Modelling_lendingclub.ipynb`)
   - Logistic regression (baseline) and XGBoost
   - **SHAP** to explain which features push a prediction toward default
5. **Results explorer** (`app/explorer.py`): a Streamlit dashboard that explains my findings to non-technical readers. Pick a factor (loan term, interest rate, debt-to-income, credit score or home ownership) to see how default rates change, plus the model results and SHAP chart in plain language.

## Results

| Model | AUC |
|---|---|
| Logistic regression | 0.713 |
| XGBoost | 0.725 |

- **Why AUC and not accuracy?** About 78.5% of loans were repaid, so a model that always predicts "repaid" already gets ~78.5% accuracy. AUC measures how well the model ranks risky borrowers above safe ones.
- **The two models are close.** Most of the signal comes from a few strong features (interest rate, loan term, debt-to-income ratio), which a simple model already captures.
- **Limitation:** at the default 0.5 cut-off, the model catches only about 12% of defaults. A lender would lower the cut-off to catch more defaults, accepting more false alarms.

## What I learned

- I originally evaluated the logistic regression on unscaled test data while training it on scaled data, which made it look much worse (AUC 0.557). Fixing this showed the simple model was nearly as good as XGBoost.
- Accuracy can be misleading on imbalanced data.

## Next steps

- Tune the decision threshold for a lender's cost of missed defaults vs. rejected good borrowers
- Try class weighting to improve recall on defaults
- Check whether the model performs equally well across loan years

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd app && streamlit run explorer.py
```

**Tools:** Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn, Streamlit
