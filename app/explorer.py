import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Credit Risk", page_icon="💳", layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #f9fafb;
        color: #1f2937;
    }

    h1, h2, h3 {
        font-weight: 700;
        color: #1e3a8a;
    }

    /* Make the two number boxes look like cards */
    div[data-testid="stMetric"] {
        background: white;
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        transition: 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
    }
    </style>
""", unsafe_allow_html=True)

st.title("Smart Credit Risk: What Makes a Loan Risky?")

st.write(
    "I looked at about 1.4 million past LendingClub loans to understand "
    "which borrowers were more likely to not pay back their loan (default). "
    "Use this page to explore what I found."
)

@st.cache_data
def load_data():
    return pd.read_csv("../data/cleaned_lendingclub_important_features.csv")

df = load_data()

st.header("The data at a glance")

col1, col2 = st.columns(2)
col1.metric("Loans analyzed", f"{len(df):,}")
col2.metric("Share that defaulted", f"{df['target_default'].mean():.1%}")

# --- Make easy-to-read groups ---
df["Loan term"] = df["term (months)"].astype(str) + " months"

df["Interest rate"] = pd.cut(
    df["int_rate"],
    bins=[0, 10, 15, 20, 25, 100],
    labels=["Under 10%", "10-15%", "15-20%", "20-25%", "25%+"],
)

df["Debt-to-income"] = pd.cut(
    df["dti"],
    bins=[-1, 10, 20, 30, 40, 1000],
    labels=["0-10", "10-20", "20-30", "30-40", "40+"],
)

df["Credit score (FICO)"] = pd.cut(
    df["fico_range_low"],
    bins=[0, 680, 700, 720, 750, 900],
    labels=["Under 680", "680-699", "700-719", "720-749", "750+"],
)

df["Home ownership"] = np.select(
    [df["home_ownership_MORTGAGE"], df["home_ownership_RENT"]],
    ["Mortgage", "Rent"],
    default="Own / Other",
)

#visual 
tab1, tab2, tab3 = st.tabs(["📊 Explore the data", "🤖 Model results", "🔍 What drives risk"])

with tab1:
    left, right = st.columns([1, 2])   # left is 1/3 wide, right is 2/3 wide

    with left:
        st.subheader("What makes a loan riskier?")
        factor = st.selectbox(
            "Pick a factor to compare:",
            ["Loan term", "Interest rate", "Debt-to-income", "Credit score (FICO)", "Home ownership"],
        )
        explanations = {
            "Loan term": "5-year (60-month) loans default about twice as often as 3-year loans.",
            "Interest rate": "The higher the interest rate, the more likely the loan is to default. Riskier borrowers usually get charged more.",
            "Debt-to-income": "People who already owe a lot compared to what they earn are more likely to default.",
            "Credit score (FICO)": "A better credit score means a lower chance of default.",
            "Home ownership": "Renters default a bit more often than people with a mortgage.",
        }
        st.info(explanations[factor])

    with right:
        default_rate = df.groupby(factor, observed=True)["target_default"].mean() * 100

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#f9fafb")   # match the page background
        ax.set_facecolor("#f9fafb")
        bars = ax.bar(default_rate.index.astype(str), default_rate.values, color="#E4572E")
        ax.bar_label(bars, fmt="%.1f%%")
        ax.set_ylabel("Default rate (%)")
        ax.set_ylim(0, default_rate.max() * 1.2)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
with tab2:
    st.header("How well can a model predict defaults?")

    results = pd.DataFrame({
        "Model": ["Logistic regression", "XGBoost"],
        "AUC": [0.713, 0.725],
    })
    st.table(results)
    
    st.write(
        "AUC measures how well a model can tell risky borrowers apart from safe ones. "
        "0.5 means it's no better than guessing, and 1.0 means perfect. "
        "Both models do a decent job, and the simpler one is almost as good, "
        "which means a few key factors (like interest rate and loan term) do most of the work."
    )

with tab3:
    st.header("What does the model pay attention to?")

    st.image("shap_summary.png", caption="SHAP summary for the XGBoost model")
    
    st.write(
        "Each row is a factor, with the most important ones at the top. "
        "Dots on the right push a loan toward default, and dots on the left push it toward being paid back. "
        "Red means a high value and blue means a low value."
    )
    
    st.write(
        "For example, high interest rates (red) sit on the right, so they raise the risk, "
        "while high credit scores (red) sit on the left, so they lower it. "
        "This matches what the charts above show."
    )
    
    st.header("The bottom line")
    st.write(
        "Interest rate, loan term, debt-to-income and credit score are the biggest warning signs. "
        "A model can help rank borrowers by risk, but it still misses many defaults, "
        "so it should support a lender's decision rather than replace it."
    )

st.divider()
st.caption(
    "Built by Cathelyn Tan · Data Science, Simon Fraser University · "
    "[GitHub](https://github.com/cathelyn06) · "
    "[LinkedIn](https://www.linkedin.com/in/cathelyn-tan)"
)