###########################################################
# AI BIAS DETECTOR – UPDATED VERSION WITH RECOMMENDATIONS
# Fully index-safe + SHAP fallback + Auto-select + PDF + Recommendations
###########################################################

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
import os
import tempfile
from io import BytesIO

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Optional models
has_xgb = False; has_lgb = False; has_cat = False; has_shap = False
try:
    import xgboost as xgb
    has_xgb = True
except:
    pass

try:
    import lightgbm as lgb
    has_lgb = True
except:
    pass

try:
    import catboost as cb
    has_cat = True
except:
    pass

try:
    import shap
    has_shap = True
except:
    pass

# PDF
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch


###########################################################
# STREAMLIT HEADER
###########################################################
st.set_page_config(page_title="AI Bias Detector", layout="wide")
st.title("AI Bias Detector – Updated Version (with PDF Recommendations)")


###########################################################
# FILE UPLOAD
###########################################################
file = st.file_uploader("Upload CSV Dataset", type=["csv"])
if file is None:
    st.stop()

df = pd.read_csv(file).reset_index(drop=True)
st.success("Dataset Loaded Successfully.")
st.dataframe(df.head())

columns = df.columns.tolist()


###########################################################
# USER INPUTS
###########################################################
target_col = st.selectbox("Select Target Column (binary)", columns)
sensitive_cols = st.multiselect("Select Sensitive Feature(s)", columns)

algo_options = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Extra Trees",
    "Gradient Boosting",
    "AdaBoost",
    "Bagging",
    "KNN",
    "Naive Bayes",
    "SVM (RBF)",
    "MLP (Neural Network)"
]

if has_xgb: algo_options.append("XGBoost")
if has_lgb: algo_options.append("LightGBM")
if has_cat: algo_options.append("CatBoost")

###########################################################
# ALGORITHM EXPLANATION TABLE (NEW SECTION)
###########################################################
st.subheader("Machine Learning Algorithms Overview")

algo_description_data = {
    "Algorithm": [
        "Logistic Regression", "Decision Tree", "Random Forest", "Extra Trees",
        "Gradient Boosting", "AdaBoost", "Bagging Classifier", "KNN",
        "Naive Bayes", "SVM (RBF)", "MLP (Neural Network)",
    ],
    "Simple Explanation": [
        "Predicts outcome using a linear boundary. Best for simple relationships.",
        "Splits data like a flowchart. Very easy to interpret.",
        "Multiple trees vote together. Reduces errors and overfitting.",
        "More random version of Random Forest. Fast and stable.",
        "Boosting method that learns from mistakes step-by-step.",
        "Focuses on misclassified points to improve accuracy.",
        "Averages many models trained on different subsets.",
        "Classifies by looking at nearest similar examples.",
        "Probability-based fast classifier assuming feature independence.",
        "Draws the best boundary between classes. Good for complex patterns.",
        "Neural network with layers. Captures nonlinear patterns.",
    ]
}

# Add optional algorithms if installed
if has_xgb:
    algo_description_data["Algorithm"].append("XGBoost")
    algo_description_data["Simple Explanation"].append(
        "Advanced boosting algorithm. Extremely powerful and popular."
    )

if has_lgb:
    algo_description_data["Algorithm"].append("LightGBM")
    algo_description_data["Simple Explanation"].append(
        "Very fast gradient boosting. Great for large datasets."
    )

if has_cat:
    algo_description_data["Algorithm"].append("CatBoost")
    algo_description_data["Simple Explanation"].append(
        "Boosting algorithm designed for categorical features."
    )

algo_df_display = pd.DataFrame(algo_description_data)

st.dataframe(
    algo_df_display,
    use_container_width=True,
    height=350
)


selected_algos = st.multiselect(
    "Select Algorithms",
    algo_options,
    default=["Random Forest", "Logistic Regression"]
)

auto_select = st.checkbox(
    "Automatically Select Best Algorithm (FAST MODE)",
    help="Tests fastest algorithms on 30% random sample."
)

na_strategy = st.selectbox("Missing Value Strategy", [
    "Drop rows with missing target/sensitive",
    "Drop any row with missing",
    "Fill NA (median/mode)"
])

test_size = st.slider("Test Size", 0.1, 0.5, 0.2)


###########################################################
# FUNCTION: BUILD MODEL
###########################################################
def build_model(name):
    name = name.lower()

    if "logistic" in name:
        return LogisticRegression(max_iter=2000)
    if "decision" in name:
        return DecisionTreeClassifier()
    if "random forest" in name:
        return RandomForestClassifier(n_estimators=200)
    if "extra" in name:
        return ExtraTreesClassifier(n_estimators=200)
    if "gradient" in name:
        return GradientBoostingClassifier()
    if "ada" in name:
        return AdaBoostClassifier()
    if "bagging" in name:
        return BaggingClassifier()
    if "knn" in name:
        return KNeighborsClassifier()
    if "naive" in name:
        return GaussianNB()
    if "svm" in name:
        return SVC(probability=True)
    if "mlp" in name:
        return MLPClassifier(max_iter=2000)
    if "xgboost" in name and has_xgb:
        return xgb.XGBClassifier(eval_metric="logloss")
    if "lightgbm" in name and has_lgb:
        return lgb.LGBMClassifier()
    if "catboost" in name and has_cat:
        return cb.CatBoostClassifier(verbose=0)

    return RandomForestClassifier()


###########################################################
# FAIRNESS
###########################################################
def compute_selection_rate(y_true, y_pred, sens_vals):
    sens_vals = np.array(sens_vals)
    groups = np.unique(sens_vals)
    out = {}
    for g in groups:
        idx = sens_vals == g
        if idx.sum() == 0:
            out[g] = np.nan
        else:
            out[g] = (y_pred[idx] == 1).mean()
    return out

def disparate_impact(sr):
    vals = [v for v in sr.values() if v == v]
    if len(vals) < 2: return np.nan
    mn, mx = min(vals), max(vals)
    if mx == 0: return np.nan
    return mn / mx

def spd(sr):
    vals = [v for v in sr.values() if v == v]
    if len(vals) < 2: return np.nan
    return max(vals) - min(vals)


###########################################################
# FEATURE IMPORTANCE
###########################################################
def compute_importance(model, X_train, X_test, y_test, use_shap, repeats):
    try:
        if use_shap and has_shap:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_test)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            vals = np.abs(shap_vals).mean(axis=0)
            return pd.DataFrame({"feature": X_train.columns, "importance": vals}), "SHAP"
    except:
        pass

    try:
        pi = permutation_importance(model, X_test, y_test, n_repeats=repeats)
        vals = pi.importances_mean
        return pd.DataFrame({"feature": X_train.columns, "importance": vals}), "Permutation Importance"
    except:
        pass

    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
        return pd.DataFrame({"feature": X_train.columns, "importance": vals}), "Feature Importances"

    return pd.DataFrame({"feature": X_train.columns, "importance": [0]*len(X_train.columns)}), "None"


###########################################################
# DATA PROCESSING
###########################################################
df2 = df.copy()

if na_strategy == "Drop rows with missing target/sensitive":
    df2 = df2.dropna(subset=[target_col] + sensitive_cols)
elif na_strategy == "Drop any row with missing":
    df2 = df2.dropna()
else:
    for c in df2.columns:
        if df2[c].dtype == "object":
            df2[c] = df2[c].fillna(df2[c].mode()[0])
        else:
            df2[c] = df2[c].fillna(df2[c].median())

df2 = df2.reset_index(drop=True)

# Encode target
y_raw = df2[target_col]
y = LabelEncoder().fit_transform(y_raw) if y_raw.dtype == "object" else y_raw.values

# Prepare X
predictors = [c for c in df2.columns if c not in [target_col] + sensitive_cols]
X = pd.get_dummies(df2[predictors], drop_first=True)

# Final split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)


###########################################################
# FAST AUTO SELECT
###########################################################
FAST_ALGOS = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Extra Trees"
]
if has_lgb: FAST_ALGOS.append("LightGBM")


###########################################################
# RUN BUTTON
###########################################################
if st.button("Run Analysis"):

    comparisons = []
    progress = st.progress(0)

    if auto_select:
        st.info("Auto-select enabled (FAST mode). Using 30% sample.")

        sample = df2.sample(frac=0.3, random_state=42).reset_index(drop=True)

        # Target
        y_s_raw = sample[target_col]
        y_s = LabelEncoder().fit_transform(y_s_raw) if y_s_raw.dtype == "object" else y_s_raw.values

        X_s = pd.get_dummies(sample[predictors], drop_first=True)

        for col in X.columns:
            if col not in X_s.columns:
                X_s[col] = 0
        X_s = X_s[X.columns]

        idx_all = np.arange(len(sample))
        idx_train, idx_test = train_test_split(
            idx_all, test_size=0.2, random_state=42, stratify=y_s
        )

        X_st = X_s.iloc[idx_train]
        X_se = X_s.iloc[idx_test]
        y_st = y_s[idx_train]
        y_se = y_s[idx_test]

        algos_to_run = FAST_ALGOS
    else:
        algos_to_run = selected_algos


    ###########################################################
    # TRAINING LOOP
    ###########################################################
    for i, algo_name in enumerate(algos_to_run):
        progress.progress(int((i / len(algos_to_run)) * 100))

        st.write(f"### Training: {algo_name}")
        model = build_model(algo_name)

        if auto_select:
            model.fit(X_st, y_st)
            preds = model.predict(X_se)
            try:
                proba = model.predict_proba(X_se)[:, 1]
                auc = roc_auc_score(y_se, proba)
            except:
                auc = np.nan

            acc = accuracy_score(y_se, preds)
            f1_val = f1_score(y_se, preds)

            # Fairness — index safe
            fairness_all = {}
            for s in sensitive_cols:
                s_series = sample[s].values
                s_test = s_series[idx_test]
                sr = compute_selection_rate(y_se, preds, s_test)
                fairness_all[s] = {
                    "selection_rates": sr,
                    "spd": spd(sr),
                    "dir": disparate_impact(sr)
                }

            imp_df, imp_method = compute_importance(
                model, X_st, X_se, y_se, False, 3
            )

        else:
            # Full data
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            try:
                proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, proba)
            except:
                auc = np.nan

            acc = accuracy_score(y_test, preds)
            f1_val = f1_score(y_test, preds)

            fairness_all = {}
            for s in sensitive_cols:
                s_series = df2[s].values
                s_test = s_series[X_test.index]
                sr = compute_selection_rate(y_test, preds, s_test)
                fairness_all[s] = {
                    "selection_rates": sr,
                    "spd": spd(sr),
                    "dir": disparate_impact(sr)
                }

            imp_df, imp_method = compute_importance(
                model, X_train, X_test, y_test, True, 10
            )

        comparisons.append({
            "algorithm": algo_name,
            "accuracy": acc,
            "f1": f1_val,
            "auc": auc,
            "fairness": fairness_all,
            "importance": imp_df,
            "imp_method": imp_method
        })

    ###########################################################
    # FINAL SUMMARY
    ###########################################################
    st.subheader("Model Comparison Summary")

    summary = pd.DataFrame([{
        "Algorithm": c["algorithm"],
        "Accuracy": c["accuracy"],
        "F1": c["f1"],
        "AUC": c["auc"],
        "Importance": c["imp_method"]
    } for c in comparisons]).sort_values("Accuracy", ascending=False)

    st.dataframe(summary)

    chosen_algo = st.selectbox("Select Algorithm to Inspect", summary["Algorithm"])
    chosen = next(c for c in comparisons if c["algorithm"] == chosen_algo)


    ###########################################################
    # FEATURE IMPORTANCE VISUALIZATION
    ###########################################################
    st.subheader(f"Feature Importance ({chosen['imp_method']}) – {chosen_algo}")

    imp_df = chosen["importance"].sort_values("importance", ascending=False)
    st.dataframe(imp_df.head(20))

    fig, ax = plt.subplots(figsize=(10, 7))
    sub = imp_df.head(20)
    ax.barh(sub["feature"][::-1], sub["importance"][::-1])
    ax.set_title("Feature Importance (Top 20)")
    ax.set_xlabel("Importance")
    st.pyplot(fig)

    # Save for PDF
    tmpdir = tempfile.mkdtemp()
    figpath = os.path.join(tmpdir, "importance.png")
    fig.savefig(figpath, bbox_inches="tight")


    ###########################################################
    # STREAMLIT RECOMMENDATIONS SECTION (NEW)
    ###########################################################
    st.subheader("Recommendations to Reduce Bias")

    top_features = imp_df.head(5)

    st.write("### Top Bias-Contributing Features")
    st.table(top_features)

    rec_html = ""

    for _, row in top_features.iterrows():
        feat = row["feature"]
        imp = row["importance"]

        rec_html += (
            f"<b>Feature:</b> {feat}<br>"
            f"<b>Influence:</b> {imp:.4f}<br>"
            f"<b>Recommendation:</b><br>"
            f"- Examine how <b>{feat}</b> differs across sensitive groups.<br>"
            f"- If imbalance exists, apply reweighting/oversampling.<br>"
            f"- Consider binning, normalization, or transformation.<br>"
            f"- If strongly bias-causing, test training model without this feature.<br><br>"
        )

    st.markdown(rec_html, unsafe_allow_html=True)


    ###########################################################
    # FAIRNESS SECTION
    ###########################################################
    st.subheader("Fairness Evaluation")

    for s in sensitive_cols:
        data = chosen["fairness"][s]
        st.write(f"### Sensitive Attribute: {s}")
        st.write("Selection Rates:", data["selection_rates"])
        st.write(f"SPD: {data['spd']:.4f}")
        st.write(f"DIR: {data['dir']:.4f}")


    ###########################################################
    # PDF GENERATION WITH RECOMMENDATIONS
    ###########################################################
    def make_pdf():

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("Bias Detection Report", styles["Title"]))
        story.append(Spacer(1, 20))

        story.append(Paragraph(f"Dataset: {file.name}", styles["Normal"]))
        story.append(Paragraph(f"Target Column: {target_col}", styles["Normal"]))
        story.append(Paragraph(f"Sensitive Columns: {', '.join(sensitive_cols)}", styles["Normal"]))
        story.append(Spacer(1, 20))


        # COMPARISON TABLE
        data = [["Algorithm", "Accuracy", "F1", "AUC", "Importance"]]
        for c in comparisons:
            data.append([
                c["algorithm"],
                f"{c['accuracy']:.3f}",
                f"{c['f1']:.3f}",
                f"{c['auc']:.3f}",
                c["imp_method"]
            ])

        table = Table(data)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.grey),
            ("GRID", (0,0), (-1,-1), 1, colors.black)
        ]))

        story.append(table)
        story.append(Spacer(1, 20))


        # FEATURE IMPORTANCE IMAGE
        story.append(Paragraph("Feature Importance", styles["Heading2"]))
        story.append(RLImage(figpath, width=6.5*inch, height=4*inch))
        story.append(Spacer(1, 20))


        # NEW! RECOMMENDATIONS IN PDF
        story.append(Paragraph("Recommendations to Reduce Bias", styles["Heading2"]))
        story.append(Spacer(1, 15))

        # TOP 5 TABLE
        top_features = imp_df.head(5)
        rec_data = [["Feature", "Importance"]]
        for _, r in top_features.iterrows():
            rec_data.append([r["feature"], f"{r['importance']:.4f}"])

        rec_table = Table(rec_data)
        rec_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("GRID", (0,0), (-1,-1), 1, colors.black),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold")
        ]))

        story.append(rec_table)
        story.append(Spacer(1, 15))

        # DETAILED RECOMMENDATIONS
        for _, row in top_features.iterrows():
            feat = row["feature"]
            imp = row["importance"]

            text = (
                f"<b>Feature:</b> {feat}<br/>"
                f"<b>Influence:</b> {imp:.4f}<br/>"
                f"<b>Recommendation:</b><br/>"
                f"- Examine how '{feat}' varies across protected groups.<br/>"
                f"- If imbalanced, apply reweighting or resampling.<br/>"
                f"- Try normalizing or transforming this feature.<br/>"
                f"- Test model performance with/without this feature.<br/><br/>"
            )

            story.append(Paragraph(text, styles["Normal"]))
            story.append(Spacer(1, 15))


        doc.build(story)
        return buffer.getvalue()


    pdf_bytes = make_pdf()

    st.download_button(
        "Download Full PDF Report",
        pdf_bytes,
        "bias_report.pdf",
        mime="application/pdf"
    )
