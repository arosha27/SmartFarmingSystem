# import streamlit as st
# import requests
# import pandas as pd
# import matplotlib.pyplot as plt

# API_URL = "http://127.0.0.1:8000"  # FastAPI backend URL

# st.title("Crop Disease Risk Prediction")

# st.markdown("Enter soil and weather parameters to predict crop disease risk.")

# # Input form
# with st.form("prediction_form"):
#     soil_pH = st.number_input("Soil pH (4.99 - 7.96)", 4.99, 7.96, step=0.01)
#     soil_moisture = st.number_input("Soil Moisture (0.056 - 0.676)", 0.056, 0.676, step=0.01)
#     soil_temp = st.number_input("Soil Temp (13.45 - 35.36 °C)", 13.45, 35.36, step=0.1)
#     nitrogen = st.number_input("Nitrogen (10.6 - 105.29)", 10.6, 105.29, step=0.1)
#     rainfall = st.number_input("Rainfall (-3.02 - 211.97 mm)", -3.02, 211.97, step=0.1)
#     humidity = st.number_input("Humidity (21.04 - 105.61 %)", 21.04, 105.61, step=0.1)
#     air_temp = st.number_input("Air Temp (10.28 - 38.78 °C)", 10.28, 38.78, step=0.1)
#     wind = st.number_input("Wind (-3.76 - 24.35 km/h)", -3.76, 24.35, step=0.1)

#     submitted = st.form_submit_button("Predict")

# if submitted:
#     input_data = {
#         "soil_pH": soil_pH,
#         "soil_moisture": soil_moisture,
#         "soil_temp": soil_temp,
#         "nitrogen": nitrogen,
#         "rainfall": rainfall,
#         "humidity": humidity,
#         "air_temp": air_temp,
#         "wind": wind
#     }

#     # Call FastAPI predict endpoint
#     response = requests.post(f"{API_URL}/predict", json=input_data)
#     explain_response = requests.post(f"{API_URL}/explain", json=input_data)

#     if response.status_code == 200 and explain_response.status_code == 200:
#         pred_data = response.json()
#         exp_data = explain_response.json()

#         st.subheader("Prediction Result")
#         st.success(f"**Risk Level:** {pred_data['prediction_label']}")

#         #Show probabilities
#         st.subheader("Prediction Probabilities")
#         probs = pred_data["probabilities"]
#         labels = ["Low Risk", "Medium Risk", "High Risk"]
#         df_probs = pd.DataFrame({"Risk Level": labels, "Probability": probs})
#         # st.bar_chart(df_probs.set_index("Risk Level") , colors=["green" , "yellow" , "red"])
        
#         plt.figure(figsize=(6,4))
#         plt.bar(df_probs["Risk Level"], df_probs["Probability"], color=["green", "yellow", "red"])
#         plt.xlabel("Risk Level")
#         plt.ylabel("Probability")
#         plt.title("Prediction Probabilities")
#         st.pyplot(plt)


#         # SHAP feature contributions
#         st.subheader("Feature Contributions")
#         feature_importance = pd.DataFrame(exp_data["feature_importance"])
#         feature_importance = feature_importance.sort_values("shap_value", ascending=True)

#         st.dataframe(feature_importance)

#         # Plot feature contributions
#         plt.figure(figsize=(8,5))
#         colors = feature_importance["shap_value"].apply(lambda x: "green" if x > 0 else "red")
#         plt.barh(feature_importance["feature"], feature_importance["shap_value"], color=colors)
#         plt.xlabel("SHAP Value (Impact on Prediction)")
#         plt.ylabel("Feature")
#         plt.title("Feature Importance for Prediction")
#         st.pyplot(plt)
        
        
        

#     else:
#         st.error("❌ Error: Could not get prediction from API")




############################## Predictions without API #####################################

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

# -------------------------
# Optional: SHAP import detection
# -------------------------
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

# -------------------------
# Cached loader for model & scaler (adjust paths if needed)
# -------------------------
MODEL_PATH = "E:/SmartFarmingSystem/models/xgboost_classifier.pickle"
SCALER_PATH = "E:/SmartFarmingSystem/models/scaler.pickle"


@st.cache_resource(show_spinner=False)
def load_model_and_scaler(model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    model_obj = None
    scaler_obj = None
    # load model
    try:
        with open(model_path, "rb") as f:
            model_obj = pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load model from '{model_path}': {e}")
    # load scaler
    try:
        with open(scaler_path, "rb") as f:
            scaler_obj = pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load scaler from '{scaler_path}': {e}")

    return model_obj, scaler_obj


model, scaler = load_model_and_scaler()

# -------------------------
# UI (kept same labels; extended ranges)
# -------------------------
st.title("Crop Disease Risk Prediction")
st.markdown("Enter soil and weather parameters to predict crop disease risk.")

with st.form("prediction_form"):
    soil_pH = st.number_input("Soil pH (4.5 - 8.5)", min_value=4.5, max_value=8.5, value=6.5, step=0.01)
    soil_moisture = st.number_input("Soil Moisture (0.0 - 0.8)", min_value=0.0, max_value=0.8, value=0.2, step=0.01)
    soil_temp = st.number_input("Soil Temp (10 - 40 °C)", min_value=10.0, max_value=40.0, value=25.0, step=0.1)
    nitrogen = st.number_input("Nitrogen (0 - 120)", min_value=0.0, max_value=120.0, value=30.0, step=0.1)
    rainfall = st.number_input("Rainfall (0 - 250 mm)", min_value=0.0, max_value=250.0, value=10.0, step=0.1)
    humidity = st.number_input("Humidity (10 - 110 %)", min_value=10.0, max_value=110.0, value=60.0, step=0.1)
    air_temp = st.number_input("Air Temp (5 - 45 °C)", min_value=5.0, max_value=45.0, value=24.0, step=0.1)
    wind = st.number_input("Wind (0 - 30 km/h)", min_value=0.0, max_value=30.0, value=3.0, step=0.1)

    submitted = st.form_submit_button("Predict")

# consistent features order used for scaler & model
FEATURE_NAMES = ["soil_pH", "soil_moisture", "soil_temp", "nitrogen", "rainfall", "humidity", "air_temp", "wind"]

# -------------------------
# Helper: robust extraction of SHAP row
# -------------------------
def _extract_shap_row(raw, class_index, n_features, model_obj=None):
    """
    Accepts shap output (list, ndarray, Explanation) and returns a 1D array length n_features
    representing shap values for the selected class_index for the first sample.
    """
    try:
        arr = np.array(raw)
    except Exception:
        # Fallback: try to convert Explanation objects with .values attribute
        try:
            if hasattr(raw, "values"):
                arr = np.array(raw.values)
            else:
                arr = np.array([])
        except Exception:
            arr = np.array([])

    try:
        # handle list[class][sample, features]
        if isinstance(raw, list):
            sel = raw[class_index] if class_index < len(raw) else raw[0]
            sel = np.array(sel)
            if sel.ndim >= 2:
                return sel[0, :n_features].astype(float)
            else:
                return sel.flatten()[:n_features].astype(float)

        if arr.size == 0:
            return np.zeros(n_features, dtype=float)

        if arr.ndim == 3:
            if arr.shape[-1] == n_features:
                # try common layouts
                if arr.shape[0] == 1 and arr.shape[1] > 1:
                    return arr[0, class_index, :].astype(float)
                if arr.shape[1] == 1 and arr.shape[0] > 1:
                    return arr[class_index, 0, :].astype(float)
                # default: take first sample of detected features axis
                return arr.reshape(-1)[0:n_features].astype(float)
            elif arr.shape[1] == n_features:
                return arr[0, :n_features, 0].astype(float)
            else:
                return arr.reshape(-1)[0:n_features].astype(float)

        if arr.ndim == 2:
            if arr.shape[1] >= n_features:
                return arr[0, :n_features].astype(float)
            else:
                return arr.flatten()[:n_features].astype(float)

        if arr.ndim == 1:
            return arr[:n_features].astype(float)

        return arr.flatten()[:n_features].astype(float)
    except Exception:
        flattened = arr.flatten()
        out = np.zeros(n_features, dtype=float)
        length = min(len(flattened), n_features)
        out[:length] = flattened[:length]
        return out

# -------------------------
# Helper: compute contributions (SHAP primary, fallback dynamic approx)
# -------------------------
def compute_feature_contributions(model_obj, scaler_obj, input_scaled, class_index, feature_names):
    n_features = len(feature_names)
    feature_importance = pd.DataFrame({"feature": feature_names, "shap_value": [0.0] * n_features})

    # Try SHAP if available and a model was loaded
    if SHAP_AVAILABLE and model_obj is not None:
        try:
            # prefer TreeExplainer for tree models
            is_xgb = model_obj.__class__.__module__.startswith("xgboost") or model_obj.__class__.__name__.lower().startswith("xgb")
            is_tree_like = hasattr(model_obj, "feature_importances_") and not hasattr(model_obj, "coef_")
            if is_xgb or is_tree_like:
                explainer = shap.TreeExplainer(model_obj)
                raw = explainer.shap_values(input_scaled)
            else:
                # small background
                try:
                    masker = shap.maskers.Independent(np.zeros((1, n_features)))
                    explainer = shap.Explainer(model_obj, masker=masker)
                    raw = explainer(input_scaled)
                except Exception:
                    explainer = shap.Explainer(model_obj)
                    raw = explainer(input_scaled)

            shap_row = _extract_shap_row(raw, class_index, n_features, model_obj)
            feature_importance = pd.DataFrame({"feature": feature_names, "shap_value": shap_row})
            feature_importance = feature_importance.sort_values("shap_value", ascending=True)
            return feature_importance
        except Exception as e_shap:
            st.warning(f"SHAP calculation failed ({e_shap}). Using fallback importance approximation.")

    # Fallback dynamic approximation using model importances / coefficients
    try:
        if model_obj is not None and hasattr(model_obj, "feature_importances_"):
            fi = np.asarray(model_obj.feature_importances_, dtype=float)
            if fi.size != n_features:
                fi = np.resize(fi, n_features)
            scaled_flat = np.asarray(input_scaled).reshape(-1)[:n_features].astype(float)
            approx = fi * scaled_flat
            feature_importance = pd.DataFrame({"feature": feature_names, "shap_value": approx})
            feature_importance = feature_importance.sort_values("shap_value", ascending=True)
            return feature_importance

        if model_obj is not None and hasattr(model_obj, "coef_"):
            coef = np.asarray(model_obj.coef_, dtype=float)
            if coef.ndim == 2:
                # try to pick appropriate row for multiclass
                try:
                    vals = coef[class_index]
                except Exception:
                    vals = coef[0]
            else:
                vals = coef.flatten()
            if vals.size != n_features:
                vals = np.resize(vals, n_features)
            scaled_flat = np.asarray(input_scaled).reshape(-1)[:n_features].astype(float)
            approx = vals * scaled_flat
            feature_importance = pd.DataFrame({"feature": feature_names, "shap_value": approx})
            feature_importance = feature_importance.sort_values("shap_value", ascending=True)
            return feature_importance

        # fallback zeros if nothing else available
        feature_importance = pd.DataFrame({"feature": feature_names, "shap_value": [0.0] * n_features})
        return feature_importance

    except Exception:
        feature_importance = pd.DataFrame({"feature": feature_names, "shap_value": [0.0] * n_features})
        return feature_importance


# -------------------------
# Prediction logic
# -------------------------
if submitted:
    # Validate model & scaler loaded
    if model is None or scaler is None:
        st.error("Model or scaler not loaded. Check the paths and server logs.")
        st.stop()

    # build raw input dict
    input_data = {
        "soil_pH": soil_pH,
        "soil_moisture": soil_moisture,
        "soil_temp": soil_temp,
        "nitrogen": nitrogen,
        "rainfall": rainfall,
        "humidity": humidity,
        "air_temp": air_temp,
        "wind": wind,
    }

    # DataFrame in consistent column order
    input_df = pd.DataFrame([[input_data[k] for k in FEATURE_NAMES]], columns=FEATURE_NAMES)

    # Scale
    try:
        input_scaled = scaler.transform(input_df)
    except Exception as e:
        st.error(f"Error while scaling input: {e}")
        st.stop()

    # Predict
    try:
        pred_raw = model.predict(input_scaled)
        pred = pred_raw[0] if hasattr(pred_raw, "__len__") else pred_raw
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        st.stop()

    # Probabilities (best-effort)
    try:
        probs = model.predict_proba(input_scaled)[0]
    except Exception:
        # fallback: try to produce a one-hot like vector
        try:
            if hasattr(model, "classes_"):
                ncls = len(model.classes_)
                probs = np.zeros(ncls, dtype=float)
                # find index for predicted class
                try:
                    idx = list(model.classes_).index(pred)
                except Exception:
                    # try integer cast fallback
                    idx = int(pred) if isinstance(pred, (int, np.integer)) and int(pred) < ncls else 0
                probs[idx] = 1.0
            else:
                # default 3-class fallback
                probs = np.array([0.0, 0.0, 0.0], dtype=float)
                try:
                    probs[int(pred)] = 1.0
                except Exception:
                    probs[0] = 1.0
        except Exception:
            probs = np.array([0.0, 0.0, 0.0], dtype=float)

    # Map class -> human label
    LABELS = ["Low Risk", "Medium Risk", "High Risk"]
    try:
        if hasattr(model, "classes_"):
            try:
                class_index = list(model.classes_).index(pred)
            except Exception:
                class_index = int(pred) if isinstance(pred, (int, np.integer)) else 0
        else:
            class_index = int(pred) if isinstance(pred, (int, np.integer)) else 0
    except Exception:
        class_index = 0

    prediction_label = LABELS[class_index] if class_index < len(LABELS) else str(pred)

    # Save prediction row to CSV
    try:
        log_entry = input_df.copy()
        log_entry["prediction"] = prediction_label
        for i, lbl in enumerate(LABELS[: len(probs)]):
            # ensure float
            log_entry[f"prob_{lbl}"] = float(probs[i])
        log_file = "predictions_log.csv"
        if os.path.exists(log_file):
            log_entry.to_csv(log_file, mode="a", header=False, index=False)
        else:
            log_entry.to_csv(log_file, mode="w", header=True, index=False)
    except Exception as e:
        st.warning(f"Could not save prediction log: {e}")

    # -------------------------
    # Show results
    # -------------------------
    st.subheader("Prediction Result")
    st.success(f"**Risk Level:** {prediction_label}")

    # Probabilities plot
    st.subheader("Prediction Probabilities")
    df_probs = pd.DataFrame({"Risk Level": LABELS[: len(probs)], "Probability": probs})
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    # Pick colors defensively (slice to match length)
    bar_colors = ["green", "yellow", "red"][: len(probs)]
    ax1.bar(df_probs["Risk Level"], df_probs["Probability"], color=bar_colors)
    ax1.set_xlabel("Risk Level")
    ax1.set_ylabel("Probability")
    ax1.set_ylim(0, 1)
    ax1.set_title("Prediction Probabilities")
    st.pyplot(fig1)
    plt.close(fig1)

    # -------------------------
    # Feature contributions (SHAP primary; dynamic fallback)
    # -------------------------
    st.subheader("Feature Contributions")
    feat_imp_df = compute_feature_contributions(model, scaler, input_scaled, class_index, FEATURE_NAMES)

    # defensive cast to numeric
    try:
        feat_imp_df["shap_value"] = feat_imp_df["shap_value"].astype(float)
    except Exception:
        feat_imp_df["shap_value"] = feat_imp_df["shap_value"].apply(lambda x: float(x) if np.isfinite(x) else 0.0)

    st.dataframe(feat_imp_df.reset_index(drop=True))

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    # color positive vs negative contributions
    colors = ["green" if v > 0 else "red" for v in feat_imp_df["shap_value"].values]
    ax2.barh(feat_imp_df["feature"], feat_imp_df["shap_value"], color=colors)
    ax2.set_xlabel("Contribution (SHAP or Approximation)")
    ax2.set_ylabel("Feature")
    ax2.set_title("Feature Importance for Current Prediction")
    st.pyplot(fig2)
    plt.close(fig2)

    # -------------------------
    # Show saved predictions log
    # -------------------------
    st.subheader("📊 Saved Predictions Log")

    log_file = "predictions_log.csv"
    if os.path.exists(log_file):
        try:
            df_log = pd.read_csv(log_file)
            st.dataframe(df_log.tail(10))  # show last 10 entries
            st.download_button(
                label="📥 Download Full Log as CSV",
                data=df_log.to_csv(index=False),
                file_name="predictions_log.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.warning(f"Could not load saved predictions: {e}")
    else:
        st.info("No predictions have been saved yet.")

