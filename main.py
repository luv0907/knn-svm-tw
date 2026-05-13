from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	mean_absolute_error,
	mean_squared_error,
	r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR


ROOT = Path(__file__).resolve().parent


def prefer_existing_path(primary: Path, fallback: Path) -> Path:
	return primary if primary.exists() else fallback


KNN_REGRESSION_CSV = prefer_existing_path(
	ROOT / "KNN" / "regrssion" / "house_prices_deploy.csv",
	ROOT / "KNN" / "regrssion" / "house_prices.csv",
)
SVM_CLASSIFICATION_CSV = ROOT / "SVM" / "classification" / "smart_healthcare_dataset.csv"
SVM_REGRESSION_CSV = prefer_existing_path(
	ROOT / "SVM" / "Regression" / "genz_social_media_usage_200k.csv",
	ROOT / "SVM" / "Regression" / "genz_social_media_usage_1M.csv",
)
SVM_REGRESSION_FEATURES = [
	"age",
	"gender",
	"daily_usage_hours",
	"primary_platform",
	"num_platforms_used",
	"avg_session_minutes",
	"night_usage",
	"addiction_level",
	"screen_time_before_sleep",
]


st.set_page_config(
	page_title="ML Studio",
	page_icon="📊",
	layout="wide",
	initial_sidebar_state="expanded",
)

UI_VERSION = "Professional UI v1.0"


def apply_creamy_theme() -> None:
	st.markdown(
		"""
		<style>
		:root {
			--cream-0: #fffaf1;
			--cream-1: #fdf2df;
			--cream-2: #f4dfbf;
			--cream-3: #e9c895;
			--ink-0: #3b2f24;
			--ink-1: #5f4932;
		}
		.stApp {
			background:
				radial-gradient(circle at 8% 6%, rgba(255, 240, 205, 0.95), transparent 26%),
				radial-gradient(circle at 88% 12%, rgba(236, 206, 160, 0.55), transparent 26%),
				radial-gradient(circle at 56% 100%, rgba(230, 193, 142, 0.25), transparent 32%),
				linear-gradient(180deg, var(--cream-0) 0%, var(--cream-1) 48%, #f5e7d2 100%);
			color: var(--ink-0);
		}
		.block-container {
			padding-top: 1rem;
			padding-bottom: 2.3rem;
			max-width: 1180px;
		}
		.hero {
			background: linear-gradient(130deg, rgba(255,255,255,0.95), rgba(255, 242, 215, 0.82));
			border: 1px solid rgba(150, 107, 59, 0.24);
			border-radius: 30px;
			padding: 1.35rem 1.5rem;
			box-shadow: 0 22px 42px rgba(97, 69, 37, 0.14);
			position: relative;
			overflow: hidden;
		}
		.hero::after {
			content: "";
			position: absolute;
			right: -60px;
			top: -72px;
			width: 210px;
			height: 210px;
			background: radial-gradient(circle, rgba(232, 196, 142, 0.44), rgba(232, 196, 142, 0));
		}
		.hero-badge {
			display: inline-block;
			padding: 0.38rem 0.86rem;
			border-radius: 999px;
			background: linear-gradient(180deg, #f1d5ad 0%, #e3bc84 100%);
			color: #4a3016;
			font-weight: 700;
			font-size: 0.82rem;
			letter-spacing: 0.06em;
			margin-bottom: 0.55rem;
			border: 1px solid rgba(128, 89, 45, 0.3);
			box-shadow: 0 8px 20px rgba(145, 100, 45, 0.2);
		}
		.panel {
			background: linear-gradient(180deg, rgba(255,255,255,0.85), rgba(255,255,255,0.72));
			border: 1px solid rgba(154, 121, 84, 0.2);
			border-radius: 24px;
			padding: 1rem 1.02rem 0.8rem 1.02rem;
			box-shadow: 0 16px 30px rgba(117, 90, 58, 0.1);
			margin-bottom: 1rem;
			transition: transform 180ms ease, box-shadow 180ms ease;
		}
		.panel:hover {
			transform: translateY(-2px);
			box-shadow: 0 22px 34px rgba(117, 90, 58, 0.13);
		}
		.panel h3, .hero h1, .hero h2, .hero p, .panel p {
			color: var(--ink-0);
		}
		.hero h1 {
			letter-spacing: 0.02em;
			font-weight: 800;
			margin-bottom: 0.2rem;
		}
		[data-testid="stMarkdown"] h2 {
			color: var(--ink-1);
			font-weight: 760;
		}
		div[data-testid="stMetric"] {
			background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(250, 238, 219, 0.85));
			border: 1px solid rgba(154, 121, 84, 0.22);
			border-radius: 18px;
			padding: 0.48rem 0.85rem;
			box-shadow: inset 0 1px 0 rgba(255,255,255,0.7);
		}
		label, .stSelectbox label, .stNumberInput label, .stSlider label {
			color: var(--ink-1) !important;
			font-weight: 600 !important;
		}
		div[data-baseweb="select"] > div,
		div[data-baseweb="input"] > div,
		div[data-testid="stNumberInput"] input {
			background: rgba(255,255,255,0.88) !important;
			border-radius: 14px !important;
			border-color: rgba(150, 107, 59, 0.25) !important;
		}
		.small-note {
			color: #6a4f33;
			font-size: 0.95rem;
		}
		div[data-testid="stButton"] button {
			border-radius: 999px;
			border: 1px solid rgba(142, 98, 55, 0.35);
			background: linear-gradient(180deg, #ffeccc 0%, #f3d8ac 100%);
			color: #4e371f;
			font-weight: 600;
			padding: 0.42rem 0.95rem;
		}
		div[data-testid="stButton"] button:hover {
			border-color: rgba(142, 98, 55, 0.65);
			color: #3b2f24;
			box-shadow: 0 8px 18px rgba(136, 95, 46, 0.22);
		}
		div[data-testid="stExpander"] {
			background: rgba(255,255,255,0.64);
			border-radius: 16px;
			border: 1px solid rgba(154, 121, 84, 0.18);
		}
		div[data-testid="stSidebar"] {
			background: linear-gradient(180deg, rgba(248, 233, 208, 0.97) 0%, rgba(241, 216, 180, 0.97) 100%);
			border-right: 1px solid rgba(140, 99, 56, 0.2);
		}
		section[data-testid="stSidebar"] h1,
		section[data-testid="stSidebar"] h2,
		section[data-testid="stSidebar"] label,
		section[data-testid="stSidebar"] p,
		section[data-testid="stSidebar"] span {
			color: #4e361d !important;
		}
		</style>
		""",
		unsafe_allow_html=True,
	)


def one_hot_encoder() -> OneHotEncoder:
	try:
		return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
	except TypeError:
		return OneHotEncoder(handle_unknown="ignore", sparse=False)

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
	df = pd.read_csv(path).drop_duplicates().reset_index(drop=True)
	return df


def get_series(df: pd.DataFrame, column: str, default=np.nan) -> pd.Series:
	if column in df.columns:
		return df[column]
	return pd.Series([default] * len(df), index=df.index)


def parse_amount(value) -> float:
	if pd.isna(value):
		return np.nan
	text = str(value).strip()
	token = ""
	for char in text:
		if char.isdigit() or char == ".":
			token += char
		elif token:
			break
	if not token:
		return np.nan
	amount = float(token)
	lowered = text.lower()
	if "lac" in lowered:
		return amount * 100000
	if "cr" in lowered:
		return amount * 10000000
	return amount


def parse_sqft(value) -> float:
	if pd.isna(value):
		return np.nan
	text = str(value).lower()
	token = ""
	for char in text:
		if char.isdigit() or char == ".":
			token += char
		elif token:
			break
def apply_professional_theme() -> None:
	st.markdown(
		"""
		<style>
		:root {
			--primary-blue: #0d47a1;
			--secondary-blue: #1565c0;
			--light-gray: #f5f5f5;
			--border-gray: #e0e0e0;
			--text-dark: #212121;
			--text-light: #616161;
		}
		.stApp {
			background-color: #fafafa;
			color: var(--text-dark);
		}
		.block-container {
			padding-top: 1.5rem;
			padding-bottom: 2rem;
			max-width: 1200px;
		}
		.hero {
			background: #ffffff;
			border: 1px solid var(--border-gray);
			border-radius: 8px;
			padding: 2rem;
			box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
			margin-bottom: 1.5rem;
		}
		.hero-badge {
			display: inline-block;
			padding: 0.4rem 0.8rem;
			border-radius: 4px;
			background: var(--primary-blue);
			color: white;
			font-weight: 600;
			font-size: 0.8rem;
			letter-spacing: 0.05em;
			margin-bottom: 1rem;
		}
		.panel {
			background: #ffffff;
			border: 1px solid var(--border-gray);
			border-radius: 8px;
			padding: 1.2rem;
			box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
			margin-bottom: 1.2rem;
		}
		.panel h3 {
			color: var(--primary-blue);
			font-weight: 600;
			margin-bottom: 1rem;
		}
		.hero h1, .hero h2 {
			color: var(--primary-blue);
			font-weight: 700;
		}
		.hero p {
			color: var(--text-light);
		}
		[data-testid="stMarkdown"] h2 {
			color: var(--primary-blue);
			font-weight: 600;
			margin-top: 1rem;
		}
		div[data-testid="stMetric"] {
			background: #ffffff;
			border: 1px solid var(--border-gray);
			border-radius: 8px;
			padding: 1rem;
			box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
		}
		label, .stSelectbox label, .stNumberInput label, .stSlider label {
			color: var(--text-dark) !important;
			font-weight: 500 !important;
			font-size: 0.95rem !important;
		}
		div[data-baseweb="select"] > div,
		div[data-baseweb="input"] > div,
		div[data-testid="stNumberInput"] input {
			background: #ffffff !important;
			border-radius: 6px !important;
			border: 1px solid var(--border-gray) !important;
		}
		.small-note {
			color: var(--text-light);
			font-size: 0.9rem;
		}
		div[data-testid="stButton"] button {
			border-radius: 6px;
			border: none;
			background: var(--primary-blue);
			color: white;
			font-weight: 600;
			padding: 0.5rem 1rem;
			transition: background 200ms ease;
		}
		div[data-testid="stButton"] button:hover {
			background: var(--secondary-blue);
		}
		div[data-testid="stExpander"] {
			background: #ffffff;
			border-radius: 6px;
			border: 1px solid var(--border-gray);
		}
		div[data-testid="stSidebar"] {
			background: #ffffff;
			border-right: 1px solid var(--border-gray);
		}
		section[data-testid="stSidebar"] h1,
		section[data-testid="stSidebar"] h2 {
			color: var(--primary-blue) !important;
		}
		section[data-testid="stSidebar"] label,
		section[data-testid="stSidebar"] p,
		section[data-testid="stSidebar"] span {
			color: var(--text-dark) !important;
		}
		</style>
		""",
		unsafe_allow_html=True,
	)
def plot_predictions(y_test: pd.Series, predictions: np.ndarray, title: str) -> plt.Figure:
	fig, ax = plt.subplots(figsize=(5.2, 4.0))
	ax.scatter(y_test, predictions, alpha=0.5, color="#1565c0")
	min_value = min(float(np.min(y_test)), float(np.min(predictions)))
	max_value = max(float(np.max(y_test)), float(np.max(predictions)))
	ax.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="#90a4ae")
	ax.set_xlabel("Actual")
	ax.set_ylabel("Predicted")
	ax.set_title(title)
	fig.tight_layout()
	return fig


def plot_residuals(y_test: pd.Series, predictions: np.ndarray, title: str) -> plt.Figure:
	fig, ax = plt.subplots(figsize=(5.2, 4.0))
	residuals = y_test - predictions
	ax.hist(residuals, bins=24, color="#d9b38c", edgecolor="#fffaf0")
	ax.set_xlabel("Residual")
	ax.set_ylabel("Count")
	ax.set_title(title)
	fig.tight_layout()
	return fig


def build_prediction_inputs(
	source: pd.DataFrame,
	feature_columns: list[str],
	key_prefix: str,
	free_text_columns: set[str] | None = None,
) -> pd.DataFrame | None:
	free_text_columns = free_text_columns or set()
	with st.form(f"{key_prefix}_prediction_form"):
		left, right = st.columns(2)
		values: dict[str, object] = {}
		for index, column in enumerate(feature_columns):
			target_column = left if index % 2 == 0 else right
			series = source[column] if column in source.columns else pd.Series(dtype="object")
			with target_column:
				if column in free_text_columns:
					default_value = ""
					if not series.empty:
						default_value = str(series.dropna().astype(str).iloc[0]) if not series.dropna().empty else ""
					values[column] = st.text_input(column, value=default_value, key=f"{key_prefix}_{column}")
				elif pd.api.types.is_numeric_dtype(series):
					default_value = float(series.median()) if not series.empty else 0.0
					min_value = float(series.min()) if not series.empty else 0.0
					max_value = float(series.max()) if not series.empty else max(default_value + 1.0, 1.0)
					if np.isfinite(min_value) and np.isfinite(max_value) and min_value != max_value:
						values[column] = st.number_input(
							column,
							min_value=min_value,
							max_value=max_value,
							value=default_value,
							key=f"{key_prefix}_{column}",
						)
					else:
						values[column] = st.number_input(column, value=default_value, key=f"{key_prefix}_{column}")
				else:
					options = series.dropna().astype(str).unique().tolist() if not series.empty else []
					if 1 <= len(options) <= 20:
						values[column] = st.selectbox(
							column,
							options,
							index=0,
							key=f"{key_prefix}_{column}",
						)
					else:
						default_value = ""
						if not series.empty and not series.dropna().empty:
							default_value = str(series.dropna().astype(str).mode().iloc[0])
						values[column] = st.text_input(column, value=default_value, key=f"{key_prefix}_{column}")

		submitted = st.form_submit_button("Predict")

	if not submitted:
		return None
	return pd.DataFrame([values], columns=feature_columns)


def app_header() -> None:
	st.markdown(
		"""
		<div class="hero">
			<div class="hero-badge">{}</div>
			<h1>Creamy ML Concept Lab</h1>
			<p class="small-note">Interactive KNN and SVM workflows for classification and regression with focused forms, live metrics, and clear model behavior.</p>
		</div>
		""".format(UI_VERSION),
		unsafe_allow_html=True,
	)


def page_overview() -> None:
	st.markdown('<div class="panel">', unsafe_allow_html=True)
	st.subheader("Model Rooms")
	st.write("Choose a module from the sidebar. Each room trains a model, shows performance, and gives a compact prediction interface tied to the concept.")
	st.markdown('</div>', unsafe_allow_html=True)

	cols = st.columns(4)
	cards = [
		("KNN Classification", "Iris data, tuned k, PCA decision boundary"),
		("KNN Regression", "House price data from the KNN folder"),
		("SVM Classification", "Iris classification with SVC"),
		("SVM Regression", "Gen Z social media mental health regression"),
	]
	for col, (title, body) in zip(cols, cards):
		with col:
			st.markdown(
				f"""
				<div class="panel">
					<h3>{title}</h3>
					<p>{body}</p>
				</div>
				""",
				unsafe_allow_html=True,
			)


def knn_classification_page() -> None:
	st.markdown('<div class="panel">', unsafe_allow_html=True)
	st.subheader("KNN Classification")
	st.write("Iris classification with a scaler + KNN pipeline, accuracy metrics, and a PCA decision boundary.")
	st.markdown('</div>', unsafe_allow_html=True)

	iris = load_iris(as_frame=True)
	df = iris.frame.copy()
	feature_columns = iris.feature_names
	target_column = "target"

	with st.sidebar:
		st.markdown("### KNN Controls")
		k_value = st.slider("Neighbors (k)", min_value=1, max_value=25, value=5, step=2)
		test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
		random_state = st.number_input("Random state", min_value=0, value=42, step=1)

	X = df[feature_columns]
	y = df[target_column]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state), stratify=y)

	model = Pipeline(
		[
			("scale", StandardScaler()),
			("knn", KNeighborsClassifier(n_neighbors=k_value)),
		]
	)
	model.fit(X_train, y_train)
	train_predictions = model.predict(X_train)
	test_predictions = model.predict(X_test)

	render_metric_row(
		[
			("Train accuracy", accuracy_score(y_train, train_predictions), ""),
			("Test accuracy", accuracy_score(y_test, test_predictions), ""),
			("Rows", float(len(df)), ""),
		]
	)

	left, right = st.columns([1.15, 0.85])
	with left:
		st.markdown('<div class="panel">', unsafe_allow_html=True)
		st.write("Classification report")
		st.text(classification_report(y_test, test_predictions, target_names=iris.target_names))
		st.markdown('</div>', unsafe_allow_html=True)

	with right:
		st.markdown('<div class="panel">', unsafe_allow_html=True)
		st.write("Confusion matrix")
		st.pyplot(plot_confusion_matrix(confusion_matrix(y_test, test_predictions), list(iris.target_names)))
		st.markdown('</div>', unsafe_allow_html=True)

	scaled_train = model.named_steps["scale"].transform(X_train)
	pca = PCA(n_components=2, random_state=int(random_state))
	train_2d = pca.fit_transform(scaled_train)
	xx, yy = np.meshgrid(
		np.linspace(train_2d[:, 0].min() - 1, train_2d[:, 0].max() + 1, 220),
		np.linspace(train_2d[:, 1].min() - 1, train_2d[:, 1].max() + 1, 220),
	)
	grid = np.c_[xx.ravel(), yy.ravel()]
	predicted_grid = model.named_steps["knn"].predict(pca.inverse_transform(grid)).reshape(xx.shape)

	boundary_fig, axis = plt.subplots(figsize=(6.8, 5.0))
	axis.contourf(xx, yy, predicted_grid, alpha=0.28, cmap="YlOrBr")
	scatter = axis.scatter(train_2d[:, 0], train_2d[:, 1], c=y_train, cmap="viridis", edgecolor="#3b2f24", s=48)
	axis.set_xlabel("PCA Component 1")
	axis.set_ylabel("PCA Component 2")
	axis.set_title("KNN Decision Boundary")
	axis.legend(*scatter.legend_elements(), title="Class", loc="best")
	boundary_fig.tight_layout()
	st.markdown('<div class="panel">', unsafe_allow_html=True)
	st.pyplot(boundary_fig)
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<div class="panel">', unsafe_allow_html=True)
	st.write("Try a prediction")
	input_frame = build_prediction_inputs(X, feature_columns, "knn_class")
	if input_frame is not None:
		prediction = model.predict(input_frame)[0]
		probability_note = ""
		st.success(f"Predicted class: {iris.target_names[int(prediction)]}")
		if hasattr(model.named_steps["knn"], "predict_proba"):
			probs = model.predict_proba(input_frame)[0]
			probability_note = ", ".join(
				f"{iris.target_names[index]}: {probability:.3f}" for index, probability in enumerate(probs)
			)
		if probability_note:
			st.caption(probability_note)
	st.markdown('</div>', unsafe_allow_html=True)


def knn_regression_page() -> None:
	st.markdown('<div class="panel">', unsafe_allow_html=True)
	st.subheader("KNN Regression")
	st.write("House price regression using only the fields you asked for: location, Carpet Area, Furnishing, facing, Bathroom, Balcony, Car Parking, and Ownership.")
	st.markdown('</div>', unsafe_allow_html=True)

	with st.sidebar:
		st.markdown("### KNN Regression Controls")
		sample_fraction = st.slider("Training sample fraction", min_value=0.05, max_value=1.0, value=0.25, step=0.05)
		neighbors = st.slider("Neighbors", min_value=1, max_value=35, value=7, step=2)
		weights = st.selectbox("Weights", ["uniform", "distance"], index=1)
		p_value = st.selectbox("Distance metric p", [1, 2], index=1)

	df = load_csv(str(KNN_REGRESSION_CSV))
	target_column = detect_house_price_target(df)
	df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
	df = df.dropna(subset=[target_column]).reset_index(drop=True)
	sampled = df.sample(frac=sample_fraction, random_state=42).reset_index(drop=True) if sample_fraction < 1.0 else df.copy()
	house_feature_columns = ["location", "Carpet Area", "Furnishing", "facing", "Bathroom", "Balcony", "Car Parking", "Ownership"]
	house_frame = sampled[[column for column in house_feature_columns if column in sampled.columns] + [target_column]].copy()
	if "Carpet Area" in house_frame.columns:
		house_frame["Carpet Area"] = house_frame["Carpet Area"].apply(parse_sqft)
	for column in ["Bathroom", "Balcony", "Car Parking"]:
		if column in house_frame.columns:
			house_frame[column] = pd.to_numeric(house_frame[column], errors="coerce")
	X = house_frame.drop(columns=[target_column])
	y = house_frame[target_column]

	preprocessor, _, _ = build_preprocessor(X)
	model = Pipeline(
		[
			("preprocess", preprocessor),
			("knn", KNeighborsRegressor(n_neighbors=neighbors, weights=weights, p=p_value)),
		]
	)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	model.fit(X_train, y_train)
	predictions = model.predict(X_test)

	mae = mean_absolute_error(y_test, predictions)
	rmse = np.sqrt(mean_squared_error(y_test, predictions))
	r2 = r2_score(y_test, predictions)
	render_metric_row(
		[
			("MAE", mae, ""),
			("RMSE", rmse, ""),
			("R2", r2, ""),
		]
	)

	left, right = st.columns(2)
	with left:
		st.markdown('<div class="panel">', unsafe_allow_html=True)
		st.pyplot(plot_regression_scatter(y_test, predictions, "Predicted vs Actual"))
		st.markdown('</div>', unsafe_allow_html=True)
	with right:
		st.markdown('<div class="panel">', unsafe_allow_html=True)
		st.pyplot(plot_residuals(y_test, predictions, "Residual Distribution"))
		st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<div class="panel">', unsafe_allow_html=True)
	with st.expander("Preview house dataset"):
		st.dataframe(df.head(10), use_container_width=True)
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<div class="panel">', unsafe_allow_html=True)
	st.write("Try a prediction")
	free_text_columns = {"location", "Furnishing", "facing", "Ownership"}
	input_frame = build_prediction_inputs(
		house_frame,
		house_feature_columns,
		"knn_reg",
		free_text_columns=free_text_columns,
	)
	if input_frame is not None:
		prediction_frame = input_frame.copy()
		if "Carpet Area" in prediction_frame.columns:
			prediction_frame["Carpet Area"] = prediction_frame["Carpet Area"].apply(parse_sqft)
		for column in ["Bathroom", "Balcony", "Car Parking"]:
			if column in prediction_frame.columns:
				prediction_frame[column] = pd.to_numeric(prediction_frame[column], errors="coerce")
		prediction_frame = prediction_frame.reindex(columns=X.columns)
		prediction = model.predict(prediction_frame)[0]
		st.success(f"Predicted price: {prediction:,.2f}")
	st.markdown('</div>', unsafe_allow_html=True)


def svm_classification_page() -> None:
	st.markdown('<div class="panel">', unsafe_allow_html=True)
	st.subheader("SVM Classification")
	st.write("Iris classification using an SVC pipeline with scaling, accuracy metrics, and a prediction form built from Iris features only.")
	st.markdown('</div>', unsafe_allow_html=True)

	iris = load_iris(as_frame=True)
	df = iris.frame.copy()
	target_column = "target"
	feature_columns = iris.feature_names

	with st.sidebar:
		st.markdown("### SVM Classification Controls")
		kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], index=0)
		c_value = st.slider("C", min_value=0.1, max_value=20.0, value=10.0, step=0.1)
		gamma = st.selectbox("Gamma", ["scale", "auto"], index=0)
		test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.3, step=0.05)

	X = df[feature_columns]
	y = df[target_column]
	preprocessor, _, _ = build_preprocessor(X)
	model = Pipeline(
		[
			("preprocess", preprocessor),
			("svm", SVC(kernel=kernel, C=c_value, gamma=gamma, probability=True, class_weight="balanced")),
		]
	)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=100, stratify=y)
	model.fit(X_train, y_train)
	train_predictions = model.predict(X_train)
	test_predictions = model.predict(X_test)

	render_metric_row(
		[
			("Train accuracy", accuracy_score(y_train, train_predictions), ""),
			("Test accuracy", accuracy_score(y_test, test_predictions), ""),
			("Rows", float(len(df)), ""),
		]
	)

	left, right = st.columns([1.15, 0.85])
	with left:
		st.markdown('<div class="panel">', unsafe_allow_html=True)
		st.write("Classification report")
		st.text(classification_report(y_test, test_predictions, target_names=iris.target_names))
		st.markdown('</div>', unsafe_allow_html=True)
	with right:
		st.markdown('<div class="panel">', unsafe_allow_html=True)
		st.write("Confusion matrix")
		st.pyplot(plot_confusion_matrix(confusion_matrix(y_test, test_predictions), list(iris.target_names)))
		st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<div class="panel">', unsafe_allow_html=True)
	with st.expander("Preview iris dataset"):
		st.dataframe(df.head(10), use_container_width=True)
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<div class="panel">', unsafe_allow_html=True)
	st.write("Try a prediction")
	input_frame = build_prediction_inputs(df, feature_columns, "svm_class")
	if input_frame is not None:
		prediction = model.predict(input_frame)[0]
		st.success(f"Predicted Iris class: {iris.target_names[int(prediction)]}")
		if hasattr(model.named_steps["svm"], "predict_proba"):
			probability = model.predict_proba(input_frame)[0]
			st.caption(", ".join(f"{iris.target_names[index]}: {score:.3f}" for index, score in enumerate(probability)))
	st.markdown('</div>', unsafe_allow_html=True)


def svm_regression_page() -> None:
	st.markdown('<div class="panel">', unsafe_allow_html=True)
	st.subheader("SVM Regression")
	st.write("Predict `mental_health_score` from the Gen Z dataset using an SVR pipeline, following the notebook's lighter workflow.")
	st.markdown('</div>', unsafe_allow_html=True)

	with st.sidebar:
		st.markdown("### SVM Regression Controls")
		sample_fraction = st.slider("Sample fraction", min_value=0.002, max_value=0.05, value=0.01, step=0.002)
		kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], index=0)
		c_value = st.slider("C", min_value=0.1, max_value=50.0, value=10.0, step=0.1)
		epsilon = st.slider("Epsilon", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
		gamma = st.selectbox("Gamma", ["scale", "auto"], index=0)

	df = load_csv(str(SVM_REGRESSION_CSV))
	df = df.sample(frac=sample_fraction, random_state=42).reset_index(drop=True) if sample_fraction < 1.0 else df.copy()
	target_column = "mental_health_score"
	st.caption(f"Target column: {target_column}")
	feature_columns = [column for column in SVM_REGRESSION_FEATURES if column in df.columns]
	st.caption("Prediction inputs: " + ", ".join(feature_columns))
	regression_frame = df[feature_columns + [target_column]].copy()
	X = regression_frame[feature_columns]
	y = df[target_column]

	preprocessor, _, _ = build_preprocessor(X)
	model = Pipeline(
		[
			("preprocess", preprocessor),
			("svm", SVR(kernel=kernel, C=c_value, epsilon=epsilon, gamma=gamma)),
		]
	)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	model.fit(X_train, y_train)
	predictions = model.predict(X_test)

	mae = mean_absolute_error(y_test, predictions)
	rmse = np.sqrt(mean_squared_error(y_test, predictions))
	r2 = r2_score(y_test, predictions)
	render_metric_row(
		[
			("MAE", mae, ""),
			("RMSE", rmse, ""),
			("R2", r2, ""),
		]
	)

	left, right = st.columns(2)
	with left:
		st.markdown('<div class="panel">', unsafe_allow_html=True)
		st.pyplot(plot_regression_scatter(y_test, predictions, "Predicted vs Actual"))
		st.markdown('</div>', unsafe_allow_html=True)
	with right:
		st.markdown('<div class="panel">', unsafe_allow_html=True)
		st.pyplot(plot_residuals(y_test, predictions, "Residual Distribution"))
		st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<div class="panel">', unsafe_allow_html=True)
	with st.expander("Preview sampled regression data"):
		st.dataframe(regression_frame.head(10), use_container_width=True)
	st.markdown('</div>', unsafe_allow_html=True)

	st.markdown('<div class="panel">', unsafe_allow_html=True)
	st.write("Try a prediction")
	input_frame = build_prediction_inputs(regression_frame, feature_columns, "svm_reg")
	if input_frame is not None:
		prediction = model.predict(input_frame)[0]
		st.success(f"Predicted {target_column}: {prediction:.3f}")
	st.markdown('</div>', unsafe_allow_html=True)


def main() -> None:
	apply_professional_theme()
	app_header()

	with st.sidebar:
		st.title("ML Studio")
		st.caption(UI_VERSION)
		st.caption("Simple model playground for your KNN and SVM notebooks")
		page = st.radio(
			"Choose a module",
			["Overview", "KNN Classification", "KNN Regression", "SVM Classification", "SVM Regression"],
			index=0,
		)
		st.markdown("---")
		st.write("Datasets")
		st.caption(f"KNN: {KNN_REGRESSION_CSV.relative_to(ROOT)}")
		st.caption("SVM/classification/iris dataset")
		st.caption(f"SVM reg: {SVM_REGRESSION_CSV.relative_to(ROOT)}")

	if page == "Overview":
		page_overview()
	elif page == "KNN Classification":
		knn_classification_page()
	elif page == "KNN Regression":
		knn_regression_page()
	elif page == "SVM Classification":
		svm_classification_page()
	else:
		svm_regression_page()


if __name__ == "__main__":
	main()
