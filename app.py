
import os
import io
import time
import logging
import streamlit as st
import pandas as pd
import numpy as np
import pyreadstat
import altair as alt

# =====================================================
# STREAMLIT UI SETUP & SESSION STATE
# =====================================================
st.set_page_config(page_title="AgPiN & AgHiN Dashboard", layout="wide")

if "lca_buffers" not in st.session_state: st.session_state.lca_buffers = None
if "lca_summary" not in st.session_state: st.session_state.lca_summary = None
if "lca_props_dict" not in st.session_state: st.session_state.lca_props_dict = None
if "lca_cond_probs_dict" not in st.session_state: st.session_state.lca_cond_probs_dict = None
if "custom_mappings" not in st.session_state: st.session_state.custom_mappings = {}
if "est_buffers" not in st.session_state: st.session_state.est_buffers = None
if "est_level_dfs_ext" not in st.session_state: st.session_state.est_level_dfs_ext = None
if "est_level_dfs_surv" not in st.session_state: st.session_state.est_level_dfs_surv = None
if "est_classes" not in st.session_state: st.session_state.est_classes = None
if "need_buffers" not in st.session_state: st.session_state.need_buffers = None
if "level_dfs" not in st.session_state: st.session_state.level_dfs = None
if "target_class" not in st.session_state: st.session_state.target_class = None

st.title("🌾 Agricultural People in Need (AgPiN) & LCA Dashboard")
st.markdown("Upload your datasets below to run the Latent Class Analysis and generate final estimates.")

# =====================================================
# SIDEBAR CONFIGURATION
# =====================================================
st.sidebar.header("1. Global Settings")
ISO3 = st.sidebar.text_input("Country ISO3 Code", value="AFG")
ROUND = st.sidebar.text_input("Survey Round", value="R11")

# --- LBN DISCLAIMER ---
if ISO3.strip().upper() == "LBN":
    st.warning("⚠️ **Disclaimer for LBN:** Please note that what is labeled as **AgPiN** (Agricultural People in Need) in these outputs is actually **AgHiN** (Agricultural Households in Need). Because the total population frame for LBN is based on agricultural households, and one cannot assume that everyone in the household is involved in agricultural activities, we can only accurately produce AgHiN (not AgPiN) for LBN.")

st.sidebar.header("2. Upload Datasets")
survey_upload = st.sidebar.file_uploader("Upload Survey Dataset (.sav, .dta, .xlsx)", type=["sav", "dta", "xlsx", "xls"])
weight_upload = st.sidebar.file_uploader("Upload Weight Input Table (.xlsx)", type=["xlsx"])

GEO_LEVELS = [
    {"var": "adm1_name", "sheet": "adm1"},
    {"var": "adm2_name", "sheet": "adm2"},
    {"var": "strata", "sheet": "strata"}
]

# =====================================================
# CORE MATH & LCA FUNCTIONS
# =====================================================
@st.cache_data(show_spinner=False)
def load_data_cached(file_name, file_bytes):
    ext = os.path.splitext(file_name)[1].lower()
    val_labels = {}
    if ext == ".sav":
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".sav") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        df, meta = pyreadstat.read_sav(tmp_path, apply_value_formats=False)
        val_labels = meta.variable_value_labels
        os.unlink(tmp_path)
    elif ext == ".dta":
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dta") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        df, meta = pyreadstat.read_dta(tmp_path, apply_value_formats=False)
        val_labels = meta.variable_value_labels
        os.unlink(tmp_path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(io.BytesIO(file_bytes))
        val_labels = {}
    else:
        raise ValueError("Unsupported file format.")
        
    if "p_ipc3plus" in df.columns and "p_3plus" not in df.columns:
        df.rename(columns={"p_ipc3plus": "p_3plus"}, inplace=True)
        
    return df, val_labels

def create_lca_variables(df):
    df = df.copy()
    lca_vars = []
    
    if "fcg" in df.columns:
        fcg = pd.to_numeric(df["fcg"], errors="coerce")
        if not fcg.isna().all():
            df["fcg_lca3"] = fcg
            lca_vars.append("fcg_lca3")
            
    if "lcsi" in df.columns:
        lcsi = pd.to_numeric(df["lcsi"], errors="coerce")
        if not lcsi.isna().all():
            df["lcsi_lca4"] = np.nan
            df.loc[lcsi == 3, "lcsi_lca4"] = 1
            df.loc[lcsi == 2, "lcsi_lca4"] = 2
            df.loc[lcsi == 1, "lcsi_lca4"] = 3
            df.loc[lcsi == 0, "lcsi_lca4"] = 4
            lca_vars.append("lcsi_lca4")
            
    if "p_3plus" in df.columns:
        p3 = pd.to_numeric(df["p_3plus"], errors="coerce")
        if not p3.isna().all():
            df["p_3plus_lca3"] = np.nan
            df.loc[p3 > 0.75, "p_3plus_lca3"] = 1
            df.loc[(p3 > 0.25) & (p3 <= 0.75), "p_3plus_lca3"] = 2
            df.loc[p3 <= 0.25, "p_3plus_lca3"] = 3
            lca_vars.append("p_3plus_lca3")
            
    return df, lca_vars

def process_custom_indicators(df, custom_configs):
    df = df.copy()
    lca_vars = []
    for col, config in custom_configs.items():
        new_col = f"{col}_custom"
        lca_vars.append(new_col)
        if config["type"] == "Categorical":
            map_df = config["map_df"]
            mapping_dict = dict(zip(map_df["Original Value"], map_df["New Mapped Value (1=Worst)"]))
            df[new_col] = df[col].map(mapping_dict)
        else:
            try:
                cuts = [-np.inf] + [float(x.strip()) for x in config["cuts"].split(",")] + [np.inf]
                labs = [float(x.strip()) for x in config["labs"].split(",")]
                num_col = pd.to_numeric(df[col], errors="coerce")
                df[new_col] = pd.cut(num_col, bins=cuts, labels=labs, ordered=False).astype(float)
            except Exception as e:
                st.error(f"Error parsing continuous cuts/labels for {col}: {e}")
                st.stop()
    return df, lca_vars

def get_id_column(df):
    for col in ["x", "hh_id", "survey_id"]:
        if col in df.columns: return col
    return None

def prepare_data(df, lca_vars):
    id_col = get_id_column(df)
    keep_cols = lca_vars.copy()
    if id_col is not None: keep_cols = [id_col] + keep_cols
    agric_col = pd.to_numeric(df.get("hh_agricactivity", 0), errors="coerce")
    df = df[agric_col.isin([1, 2, 3])].copy()
    data_full = df[keep_cols].dropna(subset=lca_vars).copy()
    data = data_full[lca_vars].astype(int).copy()
    
    X = []
    category_labels = {}
    for var in lca_vars:
        cats = sorted(data[var].unique())
        category_labels[var] = cats
        mapping = {cat: i for i, cat in enumerate(cats)}
        X.append(data[var].map(mapping).values)
        
    if len(X) > 0:
        X = np.vstack(X).T.astype(int)
    else:
        X = np.array([])
        
    n_categories = [len(category_labels[var]) for var in lca_vars]
    return data_full, data, X, category_labels, n_categories, id_col

class FastLCA:
    def __init__(self, n_classes, n_init=50, max_iter=700, tol=1e-6, seed=123):
        self.n_classes = n_classes
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.eps = 1e-12

    def _initialize(self, p, n_categories, rng):
        pi = rng.dirichlet(np.ones(self.n_classes))
        theta = [rng.dirichlet(np.ones(c), size=self.n_classes) for c in n_categories]
        return pi, theta

    def _e_step(self, X, pi, theta):
        n, p = X.shape
        log_prob = np.zeros((n, self.n_classes))
        for k in range(self.n_classes):
            log_prob[:, k] = np.log(pi[k] + self.eps)
            for j in range(p):
                log_prob[:, k] += np.log(theta[j][k, X[:, j]] + self.eps)
        max_log = log_prob.max(axis=1, keepdims=True)
        prob = np.exp(log_prob - max_log)
        denom = prob.sum(axis=1, keepdims=True)
        posterior = prob / denom
        ll = np.sum(max_log + np.log(denom + self.eps))
        return posterior, ll

    def _m_step(self, X, posterior, n_categories):
        n, p = X.shape
        Nk = posterior.sum(axis=0)
        pi = Nk / n
        theta = []
        for j in range(p):
            probs = np.zeros((self.n_classes, n_categories[j]))
            for c in range(n_categories[j]):
                mask = X[:, j] == c
                probs[:, c] = posterior[mask, :].sum(axis=0)
            probs += self.eps
            probs = probs / probs.sum(axis=1, keepdims=True)
            theta.append(probs)
        return pi, theta

    def fit(self, X, n_categories):
        rng_main = np.random.default_rng(self.seed)
        best_ll = -np.inf
        for s in range(self.n_init):
            rng = np.random.default_rng(rng_main.integers(1, 10**9))
            pi, theta = self._initialize(X.shape[1], n_categories, rng)
            old_ll = -np.inf
            for it in range(self.max_iter):
                posterior, ll = self._e_step(X, pi, theta)
                pi, theta = self._m_step(X, posterior, n_categories)
                if abs(ll - old_ll) < self.tol: break
                old_ll = ll
            posterior, ll = self._e_step(X, pi, theta)
            if ll > best_ll:
                best_ll = ll
                self.pi_ = pi
                self.theta_ = theta
                self.posterior_ = posterior
                self.log_likelihood_ = ll
                self.n_iter_ = it + 1
        self.n_obs_ = X.shape[0]
        self.n_categories_ = n_categories
        return self
    
    def predict(self): return np.argmax(self.posterior_, axis=1)
    def n_parameters(self): return (self.n_classes - 1) + sum(self.n_classes * (c - 1) for c in self.n_categories_)
    def aic(self): return -2 * self.log_likelihood_ + 2 * self.n_parameters()
    def bic(self): return -2 * self.log_likelihood_ + np.log(self.n_obs_) * self.n_parameters()
    def entropy(self):
        n, k = self.posterior_.shape
        raw = -np.sum(self.posterior_ * np.log(self.posterior_ + self.eps))
        return 1 - raw / (n * np.log(k))

def mean_max_probability(model): return np.mean(np.max(model.posterior_, axis=1))

def metric_explanations():
    return pd.DataFrame({
        "Item": [
            "--- METRICS ---", "entropy", "mean_max_probability", "min_class_prop", "max_class_prop", 
            "Average_posterior_probability", "Hard_assignment_proportion", "BIC", "AIC", "log_likelihood", "n_parameters",
            "--- METHODOLOGY ---", "What is Latent Class Analysis (LCA)?", "Algorithm / Framework", 
            "Model Robustness & Optimization", "Model Specification", "Indicators Used"
        ],
        "Explanation / Detail": [
            "", "Measures how clearly households are separated into classes. Values closer to 1 indicate clearer separation; closer to 0 indicate overlap.",
            "Average of each household's highest posterior class probability. Higher values mean assignments are made with greater certainty.",
            "Smallest estimated class share based on average posterior probabilities.", "Largest estimated class share based on average posterior probabilities.",
            "Estimated class proportion calculated as the mean posterior probability for that class across all households.",
            "Proportion of households assigned to each class based only on their highest probability (forcing a hard choice).",
            "Bayesian Information Criterion. Lower values generally indicate better model fit while penalizing complexity.",
            "Akaike Information Criterion. Lower values generally indicate better model fit.", "Model log-likelihood. Higher values indicate better fit.",
            "Number of estimated model parameters.", "",
            "LCA is an advanced statistical method used to identify hidden (latent) groups or segments within a population based on their responses to multiple categorical indicators.",
            "The model utilizes a custom-built vectorized Python implementation (using NumPy) of the Expectation-Maximization (EM) algorithm.",
            "To ensure exceptional robustness, this script executes 50 random initializations for every single model. The convergence tolerance is strictly set to 1e-6.",
            "Multinomial Latent Class Analysis assuming 'conditional independence'. Tests 2, 3, and 4 latent classes.",
            "Default uses LCSI (4 cats), FCG (3 cats), FIES-p3plus (3 cats). Custom Mode allows mapping entirely user-defined indicators."
        ]
    })

def model_comparison(models, spec_name):
    rows = []
    for k, m in models.items():
        post = m.posterior_
        rows.append({
            "specification": spec_name, "classes": k, "log_likelihood": m.log_likelihood_,
            "n_parameters": m.n_parameters(), "AIC": m.aic(), "BIC": m.bic(),
            "entropy": m.entropy(), "mean_max_probability": mean_max_probability(m),
            "min_class_prop": post.mean(axis=0).min(), "max_class_prop": post.mean(axis=0).max()
        })
    return pd.DataFrame(rows)

def class_proportions(model):
    post = model.posterior_
    hard = model.predict()
    return pd.DataFrame({
        "Class": np.arange(1, model.n_classes + 1), "Model_weight": model.pi_,
        "Average_posterior_probability": post.mean(axis=0), "Hard_assignment_proportion": [np.mean(hard == k) for k in range(model.n_classes)]
    })

def conditional_probabilities(model, lca_vars, category_labels):
    rows = []
    for j, var in enumerate(lca_vars):
        cats = category_labels[var]
        for k in range(model.n_classes):
            row = {"Variable": var, "Class": k + 1}
            for idx, cat in enumerate(cats): row[str(cat)] = model.theta_[j][k, idx]
            rows.append(row)
    return pd.DataFrame(rows)

def posterior_table(data_full, data, model, id_col):
    out = pd.DataFrame()
    if id_col is not None: out[id_col] = data_full[id_col].values
    else: out["row_id"] = data_full.index
    for col in data.columns: out[col] = data[col].values
    out["lca_class"] = model.predict() + 1
    for k in range(model.n_classes): out[f"p_class{k+1}"] = model.posterior_[:, k]
    return out

def interpretation_hints(cond_probs):
    rows = []
    for cls in sorted(cond_probs["Class"].unique()):
        sub = cond_probs[cond_probs["Class"] == cls]
        notes = []
        for _, r in sub.iterrows():
            var = r["Variable"]
            probs = r.drop(["Variable", "Class"]).dropna().astype(float)
            if probs.empty:
                notes.append(f"{var}: N/A")
            else:
                top_cat, top_val = probs.idxmax(), probs.max()
                notes.append(f"{var}: mostly {top_cat} ({top_val:.2f})")
        rows.append({"Class": cls, "Automated interpretation hint": "; ".join(notes)})
    return pd.DataFrame(rows)

def combine_cond_probs_and_hints(cond_probs, hints):
    hints_clean = pd.DataFrame({"Variable": ["" for _ in range(len(hints))], "Class": hints["Class"], "1": hints["Automated interpretation hint"]})
    return pd.concat([cond_probs, pd.DataFrame([{}]), pd.DataFrame([{"Variable": "Automated interpretation hints", "Class": ""}]), hints_clean], ignore_index=True)

def model_recommendation(summary_all):
    best_bic = summary_all.loc[summary_all["BIC"].idxmin()]
    return [f"Lowest BIC: {best_bic['specification']} with {int(best_bic['classes'])} classes."]

def save_excel_to_buffer(summary, props, cond_probs, posteriors, hints, recommendation):
    buffer = io.BytesIO()
    cond_probs_with_hints = combine_cond_probs_and_hints(cond_probs, hints)
    
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        metric_explanations().to_excel(writer, sheet_name="Metric explanations", index=False)
        summary.to_excel(writer, sheet_name="Model comparison", index=False)
        props.to_excel(writer, sheet_name="Latent class proportions", index=False)
        cond_probs_with_hints.to_excel(writer, sheet_name="Conditional probabilities", index=False)
        posteriors.to_excel(writer, sheet_name="Posteriors", index=False)
    return buffer

# --- AGPIN / AGHIN FUNCTIONS ---

def weighted_mean(df, value_col, weight_col):
    valid = df[[value_col, weight_col]].dropna()
    if valid.empty or valid[weight_col].sum() == 0: return np.nan
    return (valid[value_col] * valid[weight_col]).sum() / valid[weight_col].sum()

def weighted_se(df, value_col, weight_col):
    valid = df[[value_col, weight_col]].dropna()
    n = len(valid)
    if n < 2 or valid[weight_col].sum() == 0: return np.nan
    w, x, w_sum = valid[weight_col], valid[value_col], valid[weight_col].sum()
    xw_bar = (x * w).sum() / w_sum
    var_w = (w**2 * (x - xw_bar)**2).sum() / (w_sum**2)
    return np.sqrt(var_w * (n / (n - 1)))

def format_excel_sheet(writer, df, sheet_name):
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    workbook, worksheet = writer.book, writer.sheets[sheet_name]
    
    fmt_header = workbook.add_format({'bold': True, 'bg_color': '#D9E1F2', 'border': 1})
    fmt_header_hl = workbook.add_format({'bold': True, 'bg_color': '#FFD966', 'border': 1})
    fmt_num = workbook.add_format({'num_format': '#,##0'})
    fmt_num_hl = workbook.add_format({'num_format': '#,##0', 'bg_color': '#FFF2CC'})
    fmt_pct = workbook.add_format({'num_format': '0.0%'})
    fmt_dec = workbook.add_format({'num_format': '0.000'})
    
    for col_num, value in enumerate(df.columns.values):
        col_name = str(value).lower()
        is_highlight = (col_name.startswith("agpin_prob_class") or col_name.startswith("agpin_hard_class"))
        if "_ci_" in col_name or "_se" in col_name: is_highlight = False
            
        worksheet.write(0, col_num, value, fmt_header_hl if is_highlight else fmt_header)
        col_width = max(len(str(value)) + 2, 12)
        
        if is_highlight: worksheet.set_column(col_num, col_num, col_width, fmt_num_hl)
        elif any(x in col_name for x in ['percent']): worksheet.set_column(col_num, col_num, col_width, fmt_pct)
        elif any(x in col_name for x in ['pop', 'n_class', '_n', 'agpin', 'aghin']): worksheet.set_column(col_num, col_num, col_width, fmt_num)
        elif any(x in col_name for x in ['prob', 'se', 'ci', 'size']): worksheet.set_column(col_num, col_num, col_width, fmt_dec)
        else: worksheet.set_column(col_num, col_num, col_width)
            
    if "Total" in df.iloc[-1].values:
        worksheet.write(df.shape[0], 0, "Total", workbook.add_format({'bold': True, 'top': 1}))
    worksheet.freeze_panes(1, 0)

def clean_weight_df(df, level, adm1_ref=None):
    notes = []
    df = df.copy()
    if 'adm1_name_en' in df.columns and 'adm1_name' not in df.columns:
        df.rename(columns={'adm1_name_en': 'adm1_name'}, inplace=True)
        notes.append(f"({level}) Renamed 'adm1_name_en' to 'adm1_name'.")
    agric_pop_col = next((c for c in df.columns if "agric" in str(c).lower() and "pop" in str(c).lower()), None)
    crop_col = next((c for c in df.columns if "crop" in str(c).lower() and "produc" in str(c).lower()), None)
    live_col = next((c for c in df.columns if "livestock" in str(c).lower() and "produc" in str(c).lower()), None)
    if not agric_pop_col:
        agric_pop_col = "Percentage of agric population"
        df[agric_pop_col] = np.nan
    for c in [agric_pop_col, crop_col, live_col]:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).replace(r'[%]', '', regex=True).replace(',', '.', regex=True), errors="coerce")
            df.loc[df[c] > 1, c] = df[c] / 100.0
    if crop_col or live_col:
        cols_to_mean = [c for c in [crop_col, live_col] if c]
        mask_missing = df[agric_pop_col].isna()
        if mask_missing.any() and cols_to_mean:
            imputed_means = df.loc[mask_missing, cols_to_mean].mean(axis=1)
            valid_imputes = imputed_means.notna()
            df.loc[mask_missing & valid_imputes, agric_pop_col] = imputed_means[valid_imputes]
            if valid_imputes.any(): notes.append(f"({level}) Averaged crop & livestock percentages to fill missing agricultural population.")
    if level in ['adm2', 'strata'] and adm1_ref is not None:
        mask_missing = df[agric_pop_col].isna()
        if mask_missing.any():
            ref_agric_col = next((c for c in adm1_ref.columns if "agric" in str(c).lower() and "pop" in str(c).lower()), None)
            if ref_agric_col:
                mapping_key = None
                if 'adm1_pcode' in df.columns and 'adm1_pcode' in adm1_ref.columns:
                    mapping_dict = adm1_ref.set_index('adm1_pcode')[ref_agric_col].to_dict()
                    df.loc[mask_missing, agric_pop_col] = df.loc[mask_missing, 'adm1_pcode'].map(mapping_dict)
                    mapping_key = 'adm1_pcode'
                elif 'adm1_name' in df.columns and 'adm1_name' in adm1_ref.columns:
                    mapping_dict = adm1_ref.set_index('adm1_name')[ref_agric_col].to_dict()
                    df.loc[mask_missing, agric_pop_col] = df.loc[mask_missing, 'adm1_name'].map(mapping_dict)
                    mapping_key = 'adm1_name'
                still_missing = df[agric_pop_col].isna().sum()
                imputed_count = mask_missing.sum() - still_missing
                if imputed_count > 0: notes.append(f"({level}) Inherited agricultural population from adm1 (using {mapping_key}) for {imputed_count} rows.")
    return df, agric_pop_col, notes

# --- COMBO GENERATOR ---

def get_need_combo(row):
    f = row.get('need_type_food', 0) == 1
    a = row.get('need_type_ag_livelihood', 0) == 1
    n = row.get('need_type_nonag_livelihood', 0) == 1
    
    if a and not f and not n: return "Agriculture only"
    if a and f and not n: return "Agriculture + Food"
    if a and not f and n: return "Agriculture + Non-Agricultural Livelihoods"
    if a and f and n: return "Agriculture + Food + Non-Agricultural Livelihoods"
    if not a and f and not n: return "Food only"
    if not a and not f and n: return "Non-Agricultural Livelihoods only"
    if not a and f and n: return "Food + Non-Agricultural Livelihoods"
    return "Other / No Need"

COMBOS = [
    "Agriculture only", "Agriculture + Food", "Agriculture + Non-Agricultural Livelihoods",
    "Agriculture + Food + Non-Agricultural Livelihoods", "Food only",
    "Non-Agricultural Livelihoods only", "Food + Non-Agricultural Livelihoods", "Other / No Need"
]

NEEDS_VARS = [
    "need_type_food", "need_type_ag_livelihood", "need_type_nonag_livelihood", "need_type_other",
    "need_crop_inputs", "need_crop_infrastructure", "need_crop_knowledge", "need_ls_feed",
    "need_ls_vet_service", "need_ls_infrastructure", "need_ls_knowledge", "need_fish_inputs",
    "need_fish_infrastructure", "need_fish_knowledge", "need_env_infra_rehab", "need_cold_storage",
    "need_marketing_supp", "need_food", "need_cash", "need_vouchers_fair", "need_other",
    "need_dk", "need_ref"
]

# =====================================================
# UI TABS
# =====================================================

tab1, tab2, tab3 = st.tabs(["Step 1: Latent Class Analysis (LCA)", "Step 2: Estimation (AgPiN/AgHiN)", "Step 3: Needs Profiling"])

with tab1:
    st.header("Run Latent Class Analysis")
    
    with st.expander("📖 Read Methodology & Details"):
        st.markdown("""
        **Methodology note:**  
        This step uses household-level survey data from the DIEM `step_2` dataset to identify groups of **agricultural households** with similar food security and livelihood stress conditions. The Latent Class Analysis (LCA) is applied only to the **agricultural subset of the dataset**, defined using the `hh_agricactivity` variable, where households reporting crop production, livestock production, or both are retained for the analysis.

        The classification is based on three core indicators:
        * **LCSI**: livelihood coping strategy index, grouped into four categories: emergency, crisis, stress, and no coping.
        * **FCG**: food consumption group, grouped into three categories: poor, borderline, and acceptable.
        * **FIES `p_3plus`**: probability of moderate or severe food insecurity, grouped into three categories: severe (> 0.75), moderate (> 0.25 & ≤ 0.75), and low (≤ 0.25).

        The analysis applies a **Latent Class Analysis (LCA)** model using the **Expectation-Maximization (EM) algorithm**. This method identifies underlying, unobserved groups of agricultural households based on their response patterns across the three indicators. Rather than assigning households using fixed cut-offs only, the model estimates the probability that each household belongs to each latent class.

        Models with **2, 3, and 4 classes** are tested and compared using statistical fit measures, classification quality, and interpretability. The final class solution should be selected based not only on model statistics, but also on whether the resulting classes are meaningful, distinguishable, and useful for describing agricultural households in need.
        """)
        
    setup_mode = st.radio("Indicator Setup Mode:", ["Default DIEM (LCSI, FCG, FIES)", "Custom Indicators (Manual Prep)"])
    custom_mappings = {}
    
    if setup_mode == "Custom Indicators (Manual Prep)":
        if not survey_upload:
            st.warning("Please upload a Survey dataset in the sidebar.")
        else:
            df_cache, val_labels = load_data_cached(survey_upload.name, survey_upload.getvalue())
            selected_cols = st.multiselect("Select indicators for LCA:", df_cache.columns.tolist())
            for col in selected_cols:
                with st.expander(f"⚙️ Configure: {col}", expanded=True):
                    var_type = st.radio("Data Type", ["Categorical (Map/Merge)", "Continuous (Bin)"], key=f"type_{col}")
                    if "Categorical" in var_type:
                        unique_vals = sorted(df_cache[col].dropna().unique())
                        labels = []
                        if col in val_labels:
                            v_dict = {float(k): v for k, v in val_labels[col].items()}
                            labels = [str(v_dict.get(float(val), "")) for val in unique_vals]
                        else:
                            labels = [""] * len(unique_vals)
                        init_map = pd.DataFrame({"Original Value": unique_vals, "Label (if available)": labels, "New Mapped Value (1=Worst)": np.arange(1, len(unique_vals)+1)})
                        edited_df = st.data_editor(init_map, key=f"map_{col}", hide_index=True)
                        custom_mappings[col] = {"type": "Categorical", "map_df": edited_df}
                    else:
                        st.info("💡 **How Binning Works:** If you enter cut-offs `0.25, 0.75` and assigned values `3, 2, 1`, values ≤ 0.25 become 3, values > 0.75 become 1.")
                        cut_str = st.text_input("Cut-off values (comma separated)", "0.25, 0.75", key=f"cut_{col}")
                        lab_str = st.text_input("Assigned Values (comma separated, 1=Worst)", "3, 2, 1", key=f"lab_{col}")
                        custom_mappings[col] = {"type": "Continuous", "cuts": cut_str, "labs": lab_str}

    if st.button("▶️ Run LCA Pipeline"):
        if not survey_upload: st.error("Please upload the Survey (.sav, .dta, or Excel) file!")
        else:
            st.session_state.custom_mappings = custom_mappings 
            start_time = time.time()
            progress_bar = st.progress(0, text="Loading Survey Data (0%)")
            try:
                df, _ = load_data_cached(survey_upload.name, survey_upload.getvalue())
                if setup_mode == "Default DIEM (LCSI, FCG, FIES)":
                    df, lca_vars = create_lca_variables(df)
                    if len(lca_vars) < 3:
                        missing = set(["lcsi_lca4", "fcg_lca3", "p_3plus_lca3"]) - set(lca_vars)
                        st.warning(f"⚠️ Some default indicators were missing from your dataset ({', '.join(missing)}). The model will run using the {len(lca_vars)} available indicators.")
                    if len(lca_vars) < 2:
                        st.error("⚠️ Error: Not enough default indicators found in the dataset to run an LCA (need at least 2). Please use Custom mode.")
                        st.stop()
                    spec_name = "Default_" + "_".join([v.split('_')[0] for v in lca_vars])
                else:
                    df, lca_vars = process_custom_indicators(df, custom_mappings)
                    spec_name = "Custom"
                
                class_range = [2, 3, 4]
                data_full, data, X, category_labels, n_categories, id_col = prepare_data(df, lca_vars)
                
                if len(data_full) == 0:
                    st.error("⚠️ Error: No valid data left for analysis. The selected indicators contain missing values (NaN) for all agricultural households in this dataset. Please check your data or custom mappings.")
                    st.stop()

                models, generated_buffers = {}, {}
                
                lca_props_dict = {}
                lca_cond_probs_dict = {}
                
                for idx, k in enumerate(class_range):
                    pct = int(10 + (idx / len(class_range)) * 70)
                    progress_bar.progress(pct, text=f"Training {k}-Class Model ({pct}%).")
                    model = FastLCA(n_classes=k, n_init=50, max_iter=700, tol=1e-6, seed=123).fit(X, n_categories)
                    models[k] = model
                    
                progress_bar.progress(85, text="Generating model summaries and files (85%)...")
                summary = model_comparison(models, spec_name)
                recommendation = model_recommendation(summary)
                st.session_state.lca_summary = summary
                
                for k, model in models.items():
                    props = class_proportions(model)
                    cond_probs = conditional_probabilities(model, lca_vars, category_labels)
                    
                    lca_props_dict[k] = props
                    lca_cond_probs_dict[k] = cond_probs
                    
                    posts = posterior_table(data_full, data, model, id_col)
                    hints = interpretation_hints(cond_probs)
                    buf = save_excel_to_buffer(summary, props, cond_probs, posts, hints, recommendation)
                    generated_buffers[f"{ISO3}_{ROUND}_LCA_{spec_name}_{k}class.xlsx"] = buf.getvalue()
                
                progress_bar.progress(100, text=f"✅ Complete! Total time: {time.time() - start_time:.1f}s")
                st.session_state.lca_buffers = generated_buffers
                st.session_state.lca_props_dict = lca_props_dict
                st.session_state.lca_cond_probs_dict = lca_cond_probs_dict
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                
    if st.session_state.lca_buffers and st.session_state.lca_summary is not None:
        st.success("LCA Pipeline generated successfully. Download your models below:")
        col1, col2 = st.columns(2)
        for i, (file_name, file_data) in enumerate(st.session_state.lca_buffers.items()):
            (col1 if i % 2 == 0 else col2).download_button(label=f"📥 Download {file_name}", data=file_data, file_name=file_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.markdown("---")
        st.markdown("### 📊 LCA Model Comparison & Diagnostics")
        st.write("Review the fit indices to choose the optimal number of classes. A lower **BIC/AIC** generally indicates a better model fit. An **Entropy** closer to 1 indicates clearer, more distinct classes.")
        
        summary_df = st.session_state.lca_summary.copy()
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("**Model Fit (BIC & AIC)**")
            melted_fit = summary_df.melt(id_vars=["classes"], value_vars=["BIC", "AIC"], var_name="Metric", value_name="Value")
            fit_chart = alt.Chart(melted_fit).mark_line(point=True).encode(
                x=alt.X('classes:O', title="Number of Classes"),
                y=alt.Y('Value:Q', scale=alt.Scale(zero=False), title="Information Criterion"),
                color='Metric:N',
                tooltip=["classes", "Metric", alt.Tooltip("Value", format=",.0f")]
            ).properties(height=300)
            st.altair_chart(fit_chart, use_container_width=True)
            
        with col_c2:
            st.markdown("**Classification Quality (Entropy)**")
            ent_chart = alt.Chart(summary_df).mark_line(point=True, color='red').encode(
                x=alt.X('classes:O', title="Number of Classes"),
                y=alt.Y('entropy:Q', scale=alt.Scale(domain=[0, 1]), title="Entropy"),
                tooltip=["classes", alt.Tooltip("entropy", format=".3f")]
            ).properties(height=300)
            st.altair_chart(ent_chart, use_container_width=True)
            
        st.markdown("---")
        st.markdown("### 🔍 Model Explorer")
        st.write("Select a model below to visually inspect the estimated class sizes and item response probabilities.")
        
        selected_k = st.selectbox("Select Model to Inspect:", sorted(st.session_state.lca_props_dict.keys()))
        
        if selected_k:
            props_df = st.session_state.lca_props_dict[selected_k].copy()
            cond_probs_df = st.session_state.lca_cond_probs_dict[selected_k].copy()
            
            c_prop, c_cond = st.columns([1, 2])
            
            with c_prop:
                st.markdown(f"**Estimated Class Sizes (Model Weights)**")
                props_df["Class_Str"] = props_df["Class"].apply(lambda x: f"Class {x}")
                pie_chart = alt.Chart(props_df).mark_arc(innerRadius=50).encode(
                    theta=alt.Theta(field="Model_weight", type="quantitative"),
                    color=alt.Color(field="Class_Str", type="nominal", legend=alt.Legend(title="Class", orient="bottom"), scale=alt.Scale(scheme='category10')),
                    tooltip=["Class_Str", alt.Tooltip("Model_weight", format=".1%")]
                ).properties(height=350)
                st.altair_chart(pie_chart, use_container_width=True)
                
            with c_cond:
                st.markdown(f"**Item Response Probabilities (Heatmap)**")
                st.info("💡 **How to read:** The color intensity shows the *probability* (0% to 100%). Darker blue = higher probability. A class with dark blue squares in Category '1' (the worst category) is highly vulnerable/food insecure.")
                
                melted_cp = cond_probs_df.melt(id_vars=["Variable", "Class"], var_name="Category", value_name="Probability").dropna()
                melted_cp["Class_Str"] = "Class " + melted_cp["Class"].astype(str)
                
                def map_cat_labels(row):
                    var = row["Variable"]
                    try:
                        cat = str(int(float(row["Category"])))
                    except:
                        cat = str(row["Category"])
                        
                    if var == "fcg_lca3": return {"1": "1: Poor", "2": "2: Borderline", "3": "3: Acceptable"}.get(cat, cat)
                    if var == "lcsi_lca4": return {"1": "1: Emergency", "2": "2: Crisis", "3": "3: Stress", "4": "4: No Coping"}.get(cat, cat)
                    if var == "p_3plus_lca3": return {"1": "1: Severe/High", "2": "2: Moderate", "3": "3: Low"}.get(cat, cat)
                    
                    if "custom" in var:
                        orig_var = var.replace("_custom", "")
                        c_maps = st.session_state.get("custom_mappings", {})
                        if orig_var in c_maps and c_maps[orig_var]["type"] == "Categorical":
                            m_df = c_maps[orig_var]["map_df"]
                            match = m_df[m_df["New Mapped Value (1=Worst)"].astype(str) == cat]
                            if not match.empty:
                                lbl = match["Label (if available)"].values[0]
                                if lbl and str(lbl).strip() != "": return f"{cat}: {lbl}"
                    return cat
                    
                melted_cp["Category_Label"] = melted_cp.apply(map_cat_labels, axis=1)
                
                base_chart = alt.Chart(melted_cp).mark_rect().encode(
                    x=alt.X('Category_Label:N', title=None, axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y('Class_Str:N', title='Latent Class', sort=alt.SortField('Class', order='ascending')),
                    color=alt.Color('Probability:Q', scale=alt.Scale(scheme='blues', domain=[0, 1])),
                    tooltip=['Variable', 'Class_Str', 'Category_Label', alt.Tooltip('Probability', format='.1%')]
                ).properties(width=150, height=150)
                
                heatmap = base_chart.facet(
                    facet=alt.Facet('Variable:N', header=alt.Header(title=None)),
                    columns=3
                ).resolve_scale(x='independent')
                
                st.altair_chart(heatmap)

with tab2:
    st.header("Calculate Final Estimates")
    with st.expander("📖 Output Details & Differences"):
        st.markdown("""
        **What is the difference between the two outputs?**
        * **Output 1 (External)**: This file calculates estimates using an **external source** (e.g., the uploaded weight input table) to define the proportion of agricultural households per administrative area.
        * **Output 2 (Survey)**: This file calculates estimates using the **DIEM survey dataset** directly. It derives the proportion of agricultural households dynamically based on the `hh_agricactivity` variable in the survey.
        """)
    posterior_upload = st.file_uploader("Upload Chosen Posterior File (.xlsx)", type=["xlsx"], key="post_upload2")
    if st.button("▶️ Run Estimation Pipeline"):
        if not survey_upload or not weight_upload or not posterior_upload:
            st.error("Please ensure Survey, Weight Table, AND Posterior files are uploaded.")
        else:
            start_time = time.time()
            progress_bar = st.progress(0, text="Loading datasets (0%)...")
            try:
                survey_df, _ = load_data_cached(survey_upload.name, survey_upload.getvalue())
                post_df = pd.read_excel(posterior_upload, sheet_name="Posteriors")
                
                agric_col_num = pd.to_numeric(survey_df.get("hh_agricactivity", 0), errors="coerce")
                survey_df["agric_hh"] = np.where(agric_col_num.isin([1, 2, 3]), 1, 0)
                agric_only_df = survey_df[survey_df["agric_hh"] == 1]
                
                xls_weight = pd.ExcelFile(io.BytesIO(weight_upload.getvalue()))
                
                adm1_ref_df = None
                if "adm1" in xls_weight.sheet_names:
                    adm1_ref_df, _, _ = clean_weight_df(pd.read_excel(xls_weight, sheet_name="adm1"), "adm1")
                
                agric_survey_stats = {}
                progress_bar.progress(20, text="Computing full survey proportions (20%)...")
                
                keep_cols_list = ["weight_final", "hh_size", "hh_agricactivity"]
                id_col = next((col for col in ["x", "hh_id", "survey_id"] if col in post_df.columns and col in survey_df.columns), None)
                if id_col: keep_cols_list.append(id_col)
                for l in GEO_LEVELS:
                    keep_cols_list.append(l["var"])
                    keep_cols_list.append(l["var"].replace("_name", "_pcode"))
                
                keep_cols = list(set([c for c in keep_cols_list if c in survey_df.columns]))
                
                for level in GEO_LEVELS:
                    g_var = level["var"]
                    pcode_col = g_var.replace("_name", "_pcode")
                    grp_cols = [g_var]
                    if pcode_col != g_var and pcode_col in survey_df.columns: grp_cols.append(pcode_col)
                        
                    if g_var in survey_df.columns:
                        props = survey_df.groupby(grp_cols, dropna=False).apply(lambda x: pd.Series({"agric_percent_survey": weighted_mean(x, "agric_hh", "weight_final"), "agric_percent_survey_se": weighted_se(x, "agric_hh", "weight_final")})).reset_index()
                        hh_sizes = agric_only_df.groupby(grp_cols, dropna=False).apply(lambda x: pd.Series({"weighted_hh_size": weighted_mean(x, "hh_size", "weight_final"), "weighted_hh_size_se": weighted_se(x, "hh_size", "weight_final")})).reset_index()
                        agric_survey_stats[g_var] = props.merge(hh_sizes, on=grp_cols, how="left")
                
                merged = post_df.merge(survey_df[keep_cols], on=id_col, how="left")
                classes = sorted(merged["lca_class"].dropna().unique().astype(int))
                for c in classes:
                    merged[f"hard_class_{c}"] = np.where(merged["lca_class"] == c, 1, 0)
                    merged[f"p80_class_{c}"] = np.where(merged[f"p_class{c}"] >= 0.80, 1, 0)
                    merged[f"p90_class_{c}"] = np.where(merged[f"p_class{c}"] >= 0.90, 1, 0)
                buf_ext, buf_surv = io.BytesIO(), io.BytesIO()
                writer_ext, writer_surv = pd.ExcelWriter(buf_ext, engine="xlsxwriter"), pd.ExcelWriter(buf_surv, engine="xlsxwriter")
                def compute_metrics(x):
                    res = {"survey_n": len(x), "weighted_n": x["weight_final"].sum()}
                    for c in classes:
                        for m_name, m_col in [("prob", f"p_class{c}"), ("hard", f"hard_class_{c}"), ("prob80", f"p80_class_{c}"), ("prob90", f"p90_class_{c}")]:
                            res[f"weighted_{m_name}_class{c}"] = weighted_mean(x, m_col, "weight_final")
                            res[f"weighted_{m_name}_class{c}_se"] = weighted_se(x, m_col, "weight_final")
                        res[f"survey_n_class{c}"] = int(x[f"hard_class_{c}"].sum())
                    return pd.Series(res)
                level_dfs_ext = {}
                level_dfs_surv = {}
                for idx, level in enumerate(GEO_LEVELS):
                    g_var, w_sheet = level["var"], level["sheet"]
                    
                    if g_var not in merged.columns: 
                        st.warning(f"⚠️ Column '{g_var}' not found in survey dataset. Skipping.")
                        continue
                    
                    if w_sheet not in xls_weight.sheet_names:
                        st.warning(f"⚠️ Sheet '{w_sheet}' not found in weight input file. Skipping {g_var} estimates.")
                        continue
                        
                    progress_bar.progress(int(30 + (idx / len(GEO_LEVELS)) * 50), text=f"Processing {g_var} Estimates...")
                    pcode_col = g_var.replace("_name", "_pcode")
                    grp_cols = [g_var]
                    if pcode_col != g_var and pcode_col in merged.columns: grp_cols.append(pcode_col) 
                    class_results = merged.groupby(grp_cols, dropna=False).apply(compute_metrics).reset_index()
                    weight_df_raw = pd.read_excel(xls_weight, sheet_name=w_sheet)
                    
                    weight_df, agric_col, w_notes = clean_weight_df(weight_df_raw, w_sheet, adm1_ref_df)
                    for note in set(w_notes): st.info(f"💡 {note}")
                    
                    use_pcode = pcode_col != g_var and pcode_col in merged.columns and pcode_col in weight_df.columns
                    merge_key = pcode_col if use_pcode else g_var
                    
                    if use_pcode: st.info(f"💡 Merging {g_var} using P-codes ({pcode_col}) to securely link weight and survey data.")
                    
                    pop_col = next((c for c in weight_df.columns if "pop_count" in str(c).lower()), None)
                    weight_df["total_pop"] = pd.to_numeric(weight_df[pop_col].astype(str).replace(r'[ ,]', '', regex=True), errors="coerce") if pop_col else np.nan
                    weight_df["agric_percent_external"] = weight_df[agric_col]
                    
                    ref_cols = [c for c in [merge_key, "total_pop", "agric_percent_external"] if c in weight_df.columns]
                    
                    results = class_results.merge(weight_df[ref_cols], on=merge_key, how="left")
                    
                    if g_var in agric_survey_stats: 
                        results = results.merge(agric_survey_stats[g_var], on=grp_cols, how="left")
                    else:
                        results["agric_percent_survey"] = np.nan
                        results["agric_percent_survey_se"] = np.nan
                        results["weighted_hh_size"] = np.nan
                        results["weighted_hh_size_se"] = np.nan
                    def apply_calculations(df, use_ext):
                        df["agric_pop"] = df["total_pop"] * (df["agric_percent_external"] if use_ext else df["agric_percent_survey"])
                        df["agric_hh_total"] = df["agric_pop"] / df["weighted_hh_size"]
                        for c in classes:
                            for m in ["prob", "hard", "prob80", "prob90"]:
                                base = f"weighted_{m}_class{c}"
                                df[f"percent_{m}_class{c}"] = df[base] 
                                df[f"{base}_ci_low"] = np.maximum(0, df[base] - 1.96 * df[f"{base}_se"])
                                df[f"{base}_ci_upp"] = np.minimum(1, df[base] + 1.96 * df[f"{base}_se"])
                                df[f"agpin_{m}_class{c}"] = df["agric_pop"] * df[base]
                                df[f"agpin_{m}_class{c}_ci_low"] = df["agric_pop"] * df[f"{base}_ci_low"]
                                df[f"agpin_{m}_class{c}_ci_upp"] = df["agric_pop"] * df[f"{base}_ci_upp"]
                                df[f"aghin_{m}_class{c}"] = df["agric_hh_total"] * df[base]
                                df[f"aghin_{m}_class{c}_ci_low"] = df["agric_hh_total"] * df[f"{base}_ci_low"]
                                df[f"aghin_{m}_class{c}_ci_upp"] = df["agric_hh_total"] * df[f"{base}_ci_upp"]
                        total_row = {col: df[col].sum() for col in ["survey_n", "weighted_n", "total_pop", "agric_pop", "agric_hh_total"] if col in df.columns}
                        total_row[g_var] = "Total"
                        for c in classes:
                            total_row[f"survey_n_class{c}"] = df[f"survey_n_class{c}"].sum()
                            for m in ["prob", "hard", "prob80", "prob90"]:
                                for pfx in ["agpin", "aghin"]:
                                    col_name = f"{pfx}_{m}_class{c}"
                                    total_row[col_name] = df[col_name].sum()
                                if total_row.get("agric_pop", 0) > 0:
                                    prop = total_row[f"agpin_{m}_class{c}"] / total_row["agric_pop"]
                                    total_row[f"weighted_{m}_class{c}"] = prop
                                    total_row[f"percent_{m}_class{c}"] = prop
                        return pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
                    df_ext = apply_calculations(results.copy(), True)
                    format_excel_sheet(writer_ext, df_ext, f"{g_var}_Estimates")
                    level_dfs_ext[g_var] = df_ext
                    df_surv = apply_calculations(results.copy(), False)
                    format_excel_sheet(writer_surv, df_surv, f"{g_var}_Estimates")
                    level_dfs_surv[g_var] = df_surv
                definitions = pd.DataFrame([
                    {"Column Name": "survey_n", "Definition": "Unweighted count of surveyed agricultural households."},
                    {"Column Name": "weighted_n", "Definition": "Weighted count of surveyed agricultural households."},
                    {"Column Name": "agric_percent_external", "Definition": "Percentage of agricultural households derived from an external source (e.g., weight input table)."},
                    {"Column Name": "agric_percent_survey", "Definition": "Weighted percentage of agricultural households derived from the survey dataset (e.g., hh_agricactivity)."},
                    {"Column Name": "weighted_hh_size", "Definition": "Weighted average household size, calculated ONLY for agricultural households using the survey dataset (e.g., hh_size)."},
                    {"Column Name": "weighted_prob_classX", "Definition": "Average posterior probability of belonging to class X (preserves classification uncertainty)."},
                    {"Column Name": "weighted_hard_classX", "Definition": "Weighted proportion of households hard-assigned to class X (forces a binary choice based on highest probability)."},
                    {"Column Name": "weighted_prob80_classX", "Definition": "Weighted proportion of households whose posterior probability of belonging to class X is 80% or greater (high certainty subset)."},
                    {"Column Name": "weighted_prob90_classX", "Definition": "Weighted proportion of households whose posterior probability of belonging to class X is 90% or greater (very high certainty subset)."},
                    {"Column Name": "*_se", "Definition": "Standard Error for the metric. Calculated using Taylor Linearization to account for the complex survey weights and variance."},
                    {"Column Name": "*_ci_low / *_ci_upp", "Definition": "95% Confidence Interval bounds for the estimate (Lower Bound / Upper Bound)."},
                    {"Column Name": "agpin_*", "Definition": "Estimated Agricultural Population in Need (Absolute number of individuals)."},
                    {"Column Name": "aghin_*", "Definition": "Estimated Agricultural Households in Need (Absolute number of households)."}
                ])
                for w in [writer_ext, writer_surv]:
                    definitions.to_excel(w, sheet_name="Data_Dictionary", index=False)
                    w.sheets["Data_Dictionary"].set_column(0, 0, 30)
                    w.sheets["Data_Dictionary"].set_column(1, 1, 120)
                writer_ext.close()
                writer_surv.close()
                progress_bar.progress(100, text=f"✅ Complete! Total time: {time.time() - start_time:.1f} seconds")
                
                st.session_state.est_buffers = {f"Output_1_External_{ISO3}_{ROUND}.xlsx": buf_ext.getvalue(), f"Output_2_Survey_{ISO3}_{ROUND}.xlsx": buf_surv.getvalue()}
                st.session_state.est_level_dfs_ext = level_dfs_ext
                st.session_state.est_level_dfs_surv = level_dfs_surv
                st.session_state.est_classes = classes
            except Exception as e: st.error(f"An error occurred: {e}")
                
    if st.session_state.est_buffers and st.session_state.est_level_dfs_ext:
        st.success("Estimates generated successfully. Download your final files below:")
        col1, col2 = st.columns(2)
        for i, (fname, fdata) in enumerate(st.session_state.est_buffers.items()):
            (col1 if i % 2 == 0 else col2).download_button(label=f"📥 Download {fname}", data=fdata, file_name=fname, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # --- STEP 2 CHARTS ---
        st.markdown("---")
        st.markdown("### 📊 Population & Vulnerability Overview")
        
        c_1, c_2, c_3 = st.columns(3)
        chart_source = c_1.radio("Data Source:", ["External Population", "Survey Population"])
        chart_metric = c_2.radio("Assignment:", ["Probability (prob)", "Hard (hard)", "High Certainty (prob80)", "Very High Certainty (prob90)"])
        chart_count = c_3.radio("Show Absolute Numbers as:", ["Individuals (AgPiN)", "Households (AgHiN)"])
        
        active_dfs = st.session_state.est_level_dfs_ext if "External" in chart_source else st.session_state.est_level_dfs_surv
        m_type = "prob" if "Probability" in chart_metric else "hard" if "Hard" in chart_metric else "prob80" if "80" in chart_metric else "prob90"
        num_prefix = "agpin" if "Individuals" in chart_count else "aghin"
        classes = st.session_state.est_classes
        
        if active_dfs:
            selected_level = st.selectbox("🗺️ Select Geographic Level to View:", ["Aggregate"] + list(active_dfs.keys()))
            
            if selected_level == "Aggregate":
                first_geo = list(active_dfs.keys())[0]
                df_plot = active_dfs[first_geo][active_dfs[first_geo][first_geo] == "Total"].copy()
            else:
                df_plot = active_dfs[selected_level][active_dfs[selected_level][selected_level] != "Total"].copy()

            if not df_plot.empty and len(df_plot) > 0:
                col_chart1, col_chart2 = st.columns(2)
                
                if selected_level == "Aggregate":
                    agg_pie_data = [{"Class": f"Class {c}", "Proportion": float(df_plot[f"percent_{m_type}_class{c}"].values[0])} for c in classes if pd.notna(df_plot[f"percent_{m_type}_class{c}"].values[0])]
                    if len(agg_pie_data) > 0:
                        pie_df = pd.DataFrame(agg_pie_data)
                        with col_chart1:
                            st.markdown(f"**Vulnerability Class Distribution ({selected_level})**")
                            pie_chart = alt.Chart(pie_df).mark_arc(innerRadius=50).encode(
                                theta=alt.Theta(field="Proportion", type="quantitative"),
                                color=alt.Color(field="Class", type="nominal", scale=alt.Scale(scheme='category10')),
                                tooltip=["Class", alt.Tooltip("Proportion", format=".1%")]
                            ).properties(height=400)
                            st.altair_chart(pie_chart, use_container_width=True)
                    else:
                        col_chart1.warning(f"Not enough data to plot the Class Distribution chart for {selected_level}.")
                else:
                    with col_chart1:
                        st.markdown(f"**Vulnerability Proportions across {selected_level}**")
                        melt_cols = [f"percent_{m_type}_class{c}" for c in classes]
                        df_geo_melt = df_plot.melt(id_vars=[selected_level], value_vars=melt_cols, var_name="Class_Raw", value_name="Proportion")
                        df_geo_melt["Class"] = df_geo_melt["Class_Raw"].apply(lambda x: f"Class {str(x).split('class')[-1]}")
                        
                        if not df_geo_melt.empty and len(df_geo_melt) > 0:
                            geo_chart = alt.Chart(df_geo_melt).mark_bar().encode(
                                x=alt.X('Proportion:Q', axis=alt.Axis(format='%')),
                                y=alt.Y(f'{selected_level}:N', title=None, sort='-x'),
                                color=alt.Color('Class:N', scale=alt.Scale(scheme='category10')),
                                tooltip=[selected_level, "Class", alt.Tooltip("Proportion", format=".1%")]
                            ).properties(height=400)
                            st.altair_chart(geo_chart, use_container_width=True)
                        else:
                            st.warning(f"Not enough data to plot the Geographic chart for {selected_level}.")
                        
                with col_chart2:
                    st.markdown(f"**Estimated {chart_count} by Class ({selected_level})**")
                    if selected_level == "Aggregate":
                        agg_bar_data = [{"Class": f"Class {c}", "Absolute Number": float(df_plot[f"{num_prefix}_{m_type}_class{c}"].values[0])} for c in classes if pd.notna(df_plot[f"{num_prefix}_{m_type}_class{c}"].values[0])]
                        if len(agg_bar_data) > 0:
                            bar_df = pd.DataFrame(agg_bar_data)
                            bar_chart = alt.Chart(bar_df).mark_bar().encode(
                                x=alt.X('Class:N', axis=alt.Axis(labelAngle=0)),
                                y=alt.Y('Absolute Number:Q'),
                                color=alt.Color('Class:N', legend=None, scale=alt.Scale(scheme='category10')),
                                tooltip=["Class", alt.Tooltip("Absolute Number", format=",.0f")]
                            ).properties(height=400)
                            st.altair_chart(bar_chart, use_container_width=True)
                        else:
                            st.warning(f"Not enough data to plot Absolute Numbers chart for {selected_level}.")
                    else:
                        melt_cols_abs = [f"{num_prefix}_{m_type}_class{c}" for c in classes]
                        df_abs_melt = df_plot.melt(id_vars=[selected_level], value_vars=melt_cols_abs, var_name="Class_Raw", value_name="Absolute Number")
                        df_abs_melt["Class"] = df_abs_melt["Class_Raw"].apply(lambda x: f"Class {str(x).split('class')[-1]}")
                        
                        if not df_abs_melt.empty and len(df_abs_melt) > 0:
                            bar_chart = alt.Chart(df_abs_melt).mark_bar().encode(
                                x=alt.X('Absolute Number:Q'),
                                y=alt.Y(f'{selected_level}:N', title=None, sort='-x'),
                                color=alt.Color('Class:N', scale=alt.Scale(scheme='category10')),
                                tooltip=[selected_level, "Class", alt.Tooltip("Absolute Number", format=",.0f")]
                            ).properties(height=400)
                            st.altair_chart(bar_chart, use_container_width=True)
                        else:
                            st.warning(f"Not enough data to plot Absolute Numbers chart for {selected_level}.")
            else:
                st.warning(f"No valid data available to plot for {selected_level}.")

with tab3:
    st.header("Needs Profiling for Targeted Class")
    
    with st.expander("📖 Step 3 Methodology"):
        st.markdown("""
        **Step 3 Methodology:** Once the final latent class model is selected, households are assigned to their most likely class (hard assignment). For the selected vulnerable class, this step calculates the weighted proportions of households reporting specific types of needs (e.g., food, agricultural livelihoods). It also evaluates the intersections of needs (e.g., Agriculture + Food) to understand compounding vulnerabilities. Finally, these proportions are multiplied by the estimated AgPiN and AgHiN populations for the selected class to provide absolute numbers of people and households requiring specific assistance.
        """)

    col_a, col_b = st.columns(2)
    target_class = st.session_state.target_class = col_a.selectbox(
        "🎯 Select the most vulnerable class to profile:", [1, 2, 3, 4], 
        index=[1, 2, 3, 4].index(st.session_state.target_class) if st.session_state.target_class else 2
    )
    pop_base = col_b.radio("Calculate absolute numbers using:", ["External Population (Weight Table)", "Survey Population"])
    posterior_upload3 = st.file_uploader("Upload Chosen Posterior File (.xlsx)", type=["xlsx"], key="post_upload3")

    if st.button("▶️ Run Needs Profiling"):
        if not survey_upload or not weight_upload or not posterior_upload3:
            st.error("Please ensure Survey, Weight Table, AND Posterior files are uploaded.")
        else:
            try:
                survey_df, _ = load_data_cached(survey_upload.name, survey_upload.getvalue())
                post_df = pd.read_excel(posterior_upload3, sheet_name="Posteriors")
                
                agric_col_num = pd.to_numeric(survey_df.get("hh_agricactivity", 0), errors="coerce")
                survey_df["agric_hh"] = np.where(agric_col_num.isin([1, 2, 3]), 1, 0)
                
                xls_weight = pd.ExcelFile(io.BytesIO(weight_upload.getvalue()))
                
                adm1_ref_df = None
                if "adm1" in xls_weight.sheet_names:
                    adm1_ref_df, _, _ = clean_weight_df(pd.read_excel(xls_weight, sheet_name="adm1"), "adm1")
                
                # --- RETROFIT OLD DATASETS ---
                if "need_food" in survey_df.columns and "need_type_food" not in survey_df.columns:
                    survey_df["need_type_food"] = (pd.to_numeric(survey_df["need_food"], errors='coerce') == 1).astype(int)
                
                ag_needs = [c for c in NEEDS_VARS if c.startswith("need_crop") or c.startswith("need_ls") or c.startswith("need_fish") or c in ["need_env_infra_rehab", "need_cold_storage", "need_marketing_supp"]]
                avail_ag = [c for c in ag_needs if c in survey_df.columns]
                if avail_ag and "need_type_ag_livelihood" not in survey_df.columns:
                    survey_df["need_type_ag_livelihood"] = (survey_df[avail_ag].apply(pd.to_numeric, errors='coerce') == 1).max(axis=1).astype(int)
                    
                nonag_vars = [c for c in ["need_cash", "need_vouchers_fair"] if c in survey_df.columns]
                if nonag_vars and "need_type_nonag_livelihood" not in survey_df.columns:
                    survey_df["need_type_nonag_livelihood"] = (survey_df[nonag_vars].apply(pd.to_numeric, errors='coerce') == 1).max(axis=1).astype(int)
                
                if "need_other" in survey_df.columns and "need_type_other" not in survey_df.columns:
                    survey_df["need_type_other"] = (pd.to_numeric(survey_df["need_other"], errors='coerce') == 1).astype(int)
                
                existing_needs = [c for c in NEEDS_VARS if c in survey_df.columns]
                if not existing_needs:
                    st.warning("No need variables found in the survey dataset! Please check variable names.")
                
                id_col = next((col for col in ["x", "hh_id", "survey_id"] if col in post_df.columns and col in survey_df.columns), None)
                
                keep_cols_list = [id_col, "weight_final", "hh_size", "agric_hh"] + [l["var"] for l in GEO_LEVELS] + existing_needs
                for l in GEO_LEVELS: keep_cols_list.append(l["var"].replace("_name", "_pcode"))
                keep_cols = list(set([c for c in keep_cols_list if c in survey_df.columns]))
                
                merged = post_df.merge(survey_df[keep_cols], on=id_col, how="left")
                merged["lca_class"] = merged["lca_class"].fillna(0).astype(int)
                
                classes = sorted(merged[merged["lca_class"] > 0]["lca_class"].unique())
                for c in classes:
                    merged[f"hard_class_{c}"] = np.where(merged["lca_class"] == c, 1, 0)
                
                df_class = merged[merged["lca_class"] == target_class].copy()
                
                if df_class.empty:
                    st.error(f"No households found assigned to Class {target_class}.")
                else:
                    df_class["need_combo"] = df_class.apply(get_need_combo, axis=1)
                    for c in COMBOS:
                        df_class[f"combo_{c}"] = (df_class["need_combo"] == c).astype(int)

                    buf_needs = io.BytesIO()
                    writer_needs = pd.ExcelWriter(buf_needs, engine="xlsxwriter")
                    level_dfs = {}

                    def process_needs_level(g_var, w_sheet=None):
                        pcode_col = g_var.replace("_name", "_pcode")
                        use_pcode = False
                        
                        if g_var == "Aggregate":
                            survey_df["Aggregate"] = "All"
                            merged["Aggregate"] = "All"
                            df_class["Aggregate"] = "All"
                            grouped = df_class.groupby("Aggregate")
                            
                            if "adm1" not in xls_weight.sheet_names:
                                return pd.DataFrame()
                                
                            weight_df = pd.read_excel(xls_weight, sheet_name="adm1")
                            weight_df, agric_col, w_notes = clean_weight_df(weight_df, "adm1")
                            
                            pop_col = next((c for c in weight_df.columns if "pop_count" in str(c).lower()), None)
                            total_pop = pd.to_numeric(weight_df[pop_col].astype(str).replace(r'[ ,]', '', regex=True), errors="coerce").sum()
                            
                            if pop_base == "External Population (Weight Table)":
                                ag_pct = weight_df[agric_col].mean()
                            else:
                                ag_pct = weighted_mean(survey_df, "agric_hh", "weight_final")
                        else:
                            grp_cols = [g_var, pcode_col] if pcode_col != g_var and pcode_col in df_class.columns else [g_var]
                            grouped = df_class.groupby(grp_cols, dropna=False)
                            
                            if w_sheet not in xls_weight.sheet_names:
                                return pd.DataFrame()
                                
                            weight_df_raw = pd.read_excel(xls_weight, sheet_name=w_sheet)
                            weight_df, agric_col, w_notes = clean_weight_df(weight_df_raw, w_sheet, adm1_ref_df)
                            
                            use_pcode = pcode_col != g_var and pcode_col in df_class.columns and pcode_col in weight_df.columns
                            if use_pcode: st.info(f"💡 Merging {g_var} Needs using P-codes ({pcode_col}).")
                            
                            pop_col = next((c for c in weight_df.columns if "pop_count" in str(c).lower()), None)

                        results = []
                        for keys, group in grouped:
                            
                            if isinstance(keys, tuple):
                                name = keys[0]
                                pcode_val = keys[1] if len(keys) > 1 else None
                            else:
                                name = keys
                                pcode_val = None
                            
                            row = {g_var: name, "survey_n_in_class": len(group)}
                            if pcode_val is not None: row[pcode_col] = pcode_val
                            
                            if g_var != "Aggregate":
                                if use_pcode and pcode_val is not None and pd.notna(pcode_val):
                                    w_row = weight_df[weight_df[pcode_col] == pcode_val]
                                else:
                                    w_row = weight_df[weight_df[g_var] == name] if g_var in weight_df.columns else pd.DataFrame()
                                    
                                if not w_row.empty:
                                    t_pop = pd.to_numeric(str(w_row[pop_col].values[0]).replace(' ', '').replace(',', ''), errors="coerce")
                                    if pop_base == "External Population (Weight Table)":
                                        a_pct = w_row[agric_col].values[0]
                                        if pd.isna(a_pct): a_pct = 0
                                    else:
                                        a_pct = weighted_mean(survey_df[survey_df[g_var] == name], "agric_hh", "weight_final") if g_var in survey_df.columns else np.nan
                                else:
                                    t_pop, a_pct = 0, 0
                            else:
                                t_pop, a_pct = total_pop, ag_pct
                                
                            ag_pop = t_pop * a_pct
                            if g_var == "Aggregate":
                                whh_size = weighted_mean(survey_df[survey_df["agric_hh"] == 1], "hh_size", "weight_final")
                            else:
                                whh_size = weighted_mean(survey_df[(survey_df["agric_hh"] == 1) & (survey_df[g_var]==name)], "hh_size", "weight_final") if g_var in survey_df.columns else np.nan
                            ag_hh = ag_pop / whh_size if whh_size else 0
                            
                            full_group = merged[merged[g_var] == name] if g_var != "Aggregate" else merged
                            class_prop = weighted_mean(full_group, f"hard_class_{target_class}", "weight_final")
                            agpin_class = ag_pop * class_prop
                            aghin_class = ag_hh * class_prop
                            
                            row["Total_AgPiN_in_Class"] = agpin_class
                            row["Total_AgHiN_in_Class"] = aghin_class
                            
                            for col in existing_needs + [f"combo_{c}" for c in COMBOS]:
                                prop = weighted_mean(group, col, "weight_final")
                                row[f"percent_{col}"] = prop
                                row[f"agpin_{col}"] = agpin_class * prop
                                row[f"aghin_{col}"] = aghin_class * prop
                                
                            results.append(row)
                        return pd.DataFrame(results)

                    df_agg = process_needs_level("Aggregate")
                    if not df_agg.empty: 
                        format_excel_sheet(writer_needs, df_agg, "Aggregate_Needs")
                        level_dfs["Aggregate"] = df_agg
                    
                    for level in GEO_LEVELS:
                        if level["var"] in df_class.columns:
                            df_lvl = process_needs_level(level["var"], level["sheet"])
                            if not df_lvl.empty: 
                                format_excel_sheet(writer_needs, df_lvl, f'{level["var"]}_Needs')
                                level_dfs[level["var"]] = df_lvl

                    writer_needs.close()
                    
                    st.session_state.need_buffers = {f"Output_3_Needs_Profile_{ISO3}_{ROUND}_Class{target_class}.xlsx": buf_needs.getvalue()}
                    st.session_state.level_dfs = level_dfs

            except Exception as e:
                st.error(f"An error occurred: {e}")
                
    if st.session_state.need_buffers and st.session_state.level_dfs:
        st.success("Needs Profile Calculated!")
        
        file_name = list(st.session_state.need_buffers.keys())[0]
        st.download_button(
            label=f"📥 Download Output 3: Needs Profile",
            data=st.session_state.need_buffers[file_name],
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        df_agg = st.session_state.level_dfs.get("Aggregate")
        
        # 1. GENERAL NEEDS OVERVIEW
        st.markdown("---")
        st.markdown("### 📊 General Needs Profile (Simple Responses)")
        st.write("This chart displays the percentage of households in the target class that selected each general need category (these are independent, multiple-choice responses, not mutually exclusive combinations).")
        
        selected_level = st.selectbox("Select Geographic Level to view General Needs:", list(st.session_state.level_dfs.keys()))
        selected_df = st.session_state.level_dfs[selected_level]
        
        gen_cols = [c for c in ["need_type_food", "need_type_ag_livelihood", "need_type_nonag_livelihood", "need_type_other"] if f"percent_{c}" in selected_df.columns]
        pct_gen_cols = [f"percent_{c}" for c in gen_cols]
        
        if pct_gen_cols and not selected_df.empty:
            melted = selected_df.melt(id_vars=[selected_level], value_vars=pct_gen_cols, var_name="Need Type", value_name="Proportion")
            
            name_map = {
                "percent_need_type_food": "Food",
                "percent_need_type_ag_livelihood": "Agriculture / Livelihood",
                "percent_need_type_nonag_livelihood": "Non-Agricultural Livelihood",
                "percent_need_type_other": "Other"
            }
            melted["Need Type"] = melted["Need Type"].map(name_map)
            
            if selected_level == "Aggregate":
                bar_chart_gen = alt.Chart(melted).mark_bar(color='#E45756').encode(
                    x=alt.X('Proportion:Q', axis=alt.Axis(format='%')),
                    y=alt.Y('Need Type:N', sort='-x', title=""),
                    tooltip=["Need Type", alt.Tooltip("Proportion", format=".1%")]
                ).properties(height=300)
                st.altair_chart(bar_chart_gen, use_container_width=True)
            else:
                bar_chart_gen = alt.Chart(melted).mark_bar().encode(
                    x=alt.X('Proportion:Q', axis=alt.Axis(format='%')),
                    y=alt.Y('Need Type:N', title=None, axis=alt.Axis(labels=False)),
                    color='Need Type:N',
                    row=alt.Row(f'{selected_level}:N', header=alt.Header(labelAngle=0, labelAlign='left')),
                    tooltip=[selected_level, "Need Type", alt.Tooltip("Proportion", format=".1%")]
                ).resolve_scale(y='independent').properties(height=60)
                st.altair_chart(bar_chart_gen, use_container_width=True)

        # 2. COMBOS AND AG-SPECIFIC
        if df_agg is not None and not df_agg.empty:
            st.markdown("---")
            st.markdown("### 📊 Aggregate Level Deep-Dive")
            col_chart1, col_chart2 = st.columns(2)
            
            pie_data = []
            for c in COMBOS:
                pct = df_agg[f"percent_combo_{c}"].values[0]
                if pd.notna(pct) and pct > 0: pie_data.append({"Combination": c, "Proportion": float(pct)})
            pie_df = pd.DataFrame(pie_data)
            
            if not pie_df.empty and len(pie_df) > 0:
                with col_chart1:
                    st.markdown(f"**Need Combinations**")
                    pie_chart = alt.Chart(pie_df).mark_arc(innerRadius=50).encode(
                        theta=alt.Theta(field="Proportion", type="quantitative"),
                        color=alt.Color(field="Combination", type="nominal", legend=alt.Legend(orient="bottom")),
                        tooltip=["Combination", alt.Tooltip("Proportion", format=".1%")]
                    ).properties(height=400)
                    st.altair_chart(pie_chart, use_container_width=True)
                    
            bar_data = []
            ag_cols = [c for c in NEEDS_VARS if c.startswith("need_crop") or c.startswith("need_ls") or c.startswith("need_fish") or c in ["need_env_infra_rehab", "need_cold_storage", "need_marketing_supp"]]
            for c in ag_cols:
                if f"percent_{c}" in df_agg.columns:
                    pct = df_agg[f"percent_{c}"].values[0]
                    if pd.notna(pct): bar_data.append({"Agricultural Need Type": c.replace("need_", "").replace("_", " ").title(), "Proportion": float(pct)})
            bar_df = pd.DataFrame(bar_data).sort_values("Proportion", ascending=False)
            
            if not bar_df.empty and len(bar_df) > 0:
                with col_chart2:
                    st.markdown(f"**Specific Agricultural Needs**")
                    bar_chart = alt.Chart(bar_df).mark_bar(color='#4C78A8').encode(
                        x=alt.X('Proportion:Q', axis=alt.Axis(format='%')),
                        y=alt.Y('Agricultural Need Type:N', sort='-x'),
                        tooltip=["Agricultural Need Type", alt.Tooltip("Proportion", format=".1%")]
                    ).properties(height=400)
                    st.altair_chart(bar_chart, use_container_width=True)
