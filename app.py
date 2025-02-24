import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
import io
from contextlib import redirect_stdout
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelMetrics:
    y_true: np.ndarray
    y_pred: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    roc_auc: float
    confusion_matrix: np.ndarray
    classification_report: Dict[str, Any]
    precision: float
    recall: float
    f1: float
    class_precision: np.ndarray
    class_recall: np.ndarray
    threshold_metrics: Dict[str, np.ndarray]


class ModelEvaluator:
    def __init__(self):
        self.setup_page_config()

    @staticmethod
    def setup_page_config():
        st.set_page_config(
            page_title="ML Model Evaluation Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def load_data(self, file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        try:
            file_type = file.name.split('.')[-1].lower()
            if file_type == 'csv':
                return pd.read_csv(file), None
            elif file_type in ['xlsx', 'xls']:
                return pd.read_excel(file), None
            return None, "Unsupported file format"
        except Exception as e:
            return None, f"Error loading file: {str(e)}"

    def execute_python_file(self, content: str) -> Tuple[Dict[str, Any], str]:
        module_globals = {}
        output = io.StringIO()
        with redirect_stdout(output):
            try:
                exec(content, module_globals)
            except Exception as e:
                return {}, f"Error executing file: {str(e)}"
        return module_globals, output.getvalue()

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)

        # Calculate threshold-dependent metrics
        precisions = []
        recalls = []
        f1_scores = []
        for threshold in np.arange(0, 1.1, 0.1):
            y_pred_binary = (y_pred >= threshold).astype(int)
            precisions.append(precision_score(y_true, y_pred_binary))
            recalls.append(recall_score(y_true, y_pred_binary))
            f1_scores.append(f1_score(y_true, y_pred_binary))

        return ModelMetrics(
            y_true=y_true,
            y_pred=y_pred,
            fpr=fpr,
            tpr=tpr,
            roc_auc=auc(fpr, tpr),
            confusion_matrix=confusion_matrix(y_true, y_pred),
            classification_report=classification_report(y_true, y_pred, output_dict=True),
            precision=precision_score(y_true, y_pred),
            recall=recall_score(y_true, y_pred),
            f1=f1_score(y_true, y_pred),
            class_precision=precision_score(y_true, y_pred, average=None),
            class_recall=recall_score(y_true, y_pred, average=None),
            threshold_metrics={
                'thresholds': np.arange(0, 1.1, 0.1),
                'precision': np.array(precisions),
                'recall': np.array(recalls),
                'f1': np.array(f1_scores)
            }
        )

    def plot_roc_curve(self, metrics: ModelMetrics):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(metrics.fpr, metrics.tpr, 'darkorange',
                label=f'ROC curve (AUC = {metrics.roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], 'navy', linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve', fontsize=14, pad=20)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

    def plot_confusion_matrix(self, metrics: ModelMetrics):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(metrics.confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix', pad=20)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        st.pyplot(fig)

    def display_classification_report(self, metrics: ModelMetrics):
        df_report = pd.DataFrame(metrics.classification_report).transpose()
        st.dataframe(
            df_report.style.format("{:.2f}")
            .background_gradient(cmap='Blues')
            .set_table_styles([
                {'selector': 'th', 'props': [('font-weight', 'bold')]},
                {'selector': 'td', 'props': [('text-align', 'center')]}
            ])
        )

    def render_metrics_dashboard(self, metrics: ModelMetrics):
        # Main metrics
        st.subheader("📊 Overall Performance Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Precision", f"{metrics.precision:.3f}")
        col2.metric("Recall", f"{metrics.recall:.3f}")
        col3.metric("F1 Score", f"{metrics.f1:.3f}")

        # ROC and Confusion Matrix
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("ROC Curve")
            self.plot_roc_curve(metrics)
        with col2:
            st.subheader("Confusion Matrix")
            self.plot_confusion_matrix(metrics)


        # Classification Report
        with st.expander("Detailed Classification Report", expanded=False):
            self.display_classification_report(metrics)

    def handle_python_file(self):
        uploaded_file = st.file_uploader("Upload Python File", type=['py'])
        if not uploaded_file:
            return

        content = uploaded_file.getvalue().decode("utf-8")
        module_globals, output = self.execute_python_file(content)

        if output:
            with st.expander("Execution Output", expanded=True):
                st.code(output)

        col1, col2 = st.columns(2)
        with col1:
            true_var = st.text_input("True labels variable:", "y_true")
        with col2:
            pred_var = st.text_input("Predicted labels variable:", "y_pred")

        if st.button('Generate Evaluation', type='primary'):
            y_true = module_globals.get(true_var)
            y_pred = module_globals.get(pred_var)

            if y_true is None or y_pred is None:
                st.error(f"Variables '{true_var}' and/or '{pred_var}' not found")
                return

            try:
                metrics = self.calculate_metrics(y_true, y_pred)
                self.render_metrics_dashboard(metrics)
            except Exception as e:
                st.error(f"Error calculating metrics: {str(e)}")

    def handle_data_file(self):
        uploaded_file = st.file_uploader("Upload Data File", type=['csv', 'xlsx', 'xls'])
        if not uploaded_file:
            return

        df, error = self.load_data(uploaded_file)
        if error:
            st.error(error)
            return

        st.dataframe(df.head())
        cols = df.columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            true_col = st.selectbox("True Labels Column", cols)
        with col2:
            pred_col = st.selectbox("Predicted Labels Column", cols)

        if st.button('Generate Evaluation', type='primary'):
            try:
                metrics = self.calculate_metrics(df[true_col], df[pred_col])
                self.render_metrics_dashboard(metrics)
            except Exception as e:
                st.error(f"Error calculating metrics: {str(e)}")

    def run(self):
        st.title('🎯 Model Evaluation Dashboard')

        with st.sidebar:
            st.header("Settings")
            file_type = st.radio(
                "Select Input Type:",
                ['Python File (.py)', 'Data File (CSV/Excel)'],
                help="Choose how you want to input your model predictions"
            )

        if file_type == 'Python File (.py)':
            self.handle_python_file()
        else:
            self.handle_data_file()


if __name__ == '__main__':
    app = ModelEvaluator()
    app.run()