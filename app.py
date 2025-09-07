import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Custom implementation of ML algorithms
from collections import Counter
import math
import random
from sklearn.datasets import load_breast_cancer

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #e91e63;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1rem 0;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #e91e63;
        margin: 0.5rem 0;
    }
    .stAlert > div {
        padding-top: 1rem;
    }
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .benign {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .malignant {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Custom implementation of train_test_split
def custom_train_test_split(X, y, test_size=0.2, random_state=None, stratify=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Create indices
    indices = np.arange(n_samples)
    
    if stratify is not None:
        # Simple stratification implementation
        unique_classes = np.unique(stratify)
        train_indices = []
        test_indices = []
        
        for cls in unique_classes:
            cls_indices = indices[stratify == cls]
            n_cls_test = int(len(cls_indices) * test_size)
            
            np.random.shuffle(cls_indices)
            test_indices.extend(cls_indices[:n_cls_test])
            train_indices.extend(cls_indices[n_cls_test:])
    else:
        np.random.shuffle(indices)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
    
    # Return splits
    X_train = X.iloc[train_indices] if hasattr(X, 'iloc') else X[train_indices]
    X_test = X.iloc[test_indices] if hasattr(X, 'iloc') else X[test_indices]
    y_train = y.iloc[train_indices] if hasattr(y, 'iloc') else y[train_indices]
    y_test = y.iloc[test_indices] if hasattr(y, 'iloc') else y[test_indices]
    
    return X_train, X_test, y_train, y_test

# Custom implementation of StandardScaler
class CustomStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # Avoid division by zero
        self.scale_[self.scale_ == 0] = 1.0
        return self
        
    def transform(self, X):
        return (X - self.mean_) / self.scale_
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)

# Custom implementation of Logistic Regression
class CustomLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None
        
    def _sigmoid(self, z):
        # Prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        # Ensure X is 2D array
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.max_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(linear_model)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict_proba(self, X):
        # Ensure X is 2D array
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        # Handle both 1D and 2D cases
        if len(proba.shape) > 1 and proba.shape[1] > 1:
            return (proba[:, 1] >= threshold).astype(int)
        else:
            return (proba >= threshold).astype(int)

# Custom implementation of K-Nearest Neighbors
class CustomKNN:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        # Ensure X is 2D array
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        # Ensure X is 2D array
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        predictions = []
        for x in X:
            # Compute distances
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            
            # Get k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            
            # Majority vote
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        # Ensure X is 2D array
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        probas = []
        for x in X:
            # Compute distances
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            
            # Get k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            
            # Calculate probabilities
            count_0 = np.sum(k_nearest_labels == 0)
            count_1 = np.sum(k_nearest_labels == 1)
            prob_0 = count_0 / self.k
            prob_1 = count_1 / self.k
            probas.append([prob_0, prob_1])
        
        return np.array(probas)

# Custom implementation of Naive Bayes
class CustomNaiveBayes:
    def __init__(self):
        self.class_priors = None
        self.class_mean = None
        self.class_var = None
        
    def fit(self, X, y):
        # Ensure X is 2D array
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Calculate mean, variance, and prior for each class
        self.class_mean = np.zeros((n_classes, n_features))
        self.class_var = np.zeros((n_classes, n_features))
        self.class_priors = np.zeros(n_classes)
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_mean[idx, :] = X_c.mean(axis=0)
            self.class_var[idx, :] = X_c.var(axis=0)
            self.class_priors[idx] = X_c.shape[0] / float(n_samples)
    
    def _pdf(self, class_idx, x):
        mean = self.class_mean[class_idx]
        var = self.class_var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var + 1e-9))  # Add small value to avoid division by zero
        denominator = np.sqrt(2 * np.pi * var + 1e-9)
        return numerator / denominator
    
    def predict_proba(self, X):
        # Ensure X is 2D array
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        probabilities = np.zeros((n_samples, n_classes))
        
        for idx, c in enumerate(self.classes):
            prior = self.class_priors[idx]
            likelihood = np.prod(self._pdf(idx, X), axis=1)
            probabilities[:, idx] = prior * likelihood
        
        # Normalize probabilities
        probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
        return probabilities
    
    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes[np.argmax(probas, axis=1)]

# Custom evaluation metrics
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

def roc_curve(y_true, y_score):
    # Sort by score
    sorted_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_score_sorted = y_score[sorted_indices]
    
    # Initialize variables
    tpr = [0]
    fpr = [0]
    thresholds = [y_score_sorted[0] + 1]
    
    # Calculate TP, FP, TN, FN for each threshold
    for i in range(len(y_score_sorted)):
        threshold = y_score_sorted[i]
        y_pred = (y_score_sorted >= threshold).astype(int)
        
        tp = np.sum((y_true_sorted == 1) & (y_pred == 1))
        fp = np.sum((y_true_sorted == 0) & (y_pred == 1))
        tn = np.sum((y_true_sorted == 0) & (y_pred == 0))
        fn = np.sum((y_true_sorted == 1) & (y_pred == 0))
        
        tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
        thresholds.append(threshold)
    
    # Add endpoint
    tpr.append(1)
    fpr.append(1)
    thresholds.append(y_score_sorted[-1] - 1)
    
    return np.array(fpr), np.array(tpr), np.array(thresholds)

def auc(x, y):
    # Calculate area under curve using trapezoidal rule
    direction = 1
    dx = np.diff(x)
    if np.any(dx < 0):
        if np.all(dx <= 0):
            direction = -1
        else:
            raise ValueError("x is neither increasing nor decreasing")
    return direction * np.trapz(y, x)

def roc_auc_score(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)

@st.cache_data
def load_data():
    """Load and prepare the breast cancer dataset"""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df['diagnosis'] = df['target'].map({0: 'Malignant', 1: 'Benign'})
    return df, data

@st.cache_data
def prepare_features(df):
    """Prepare features for modeling"""
    X = df.drop(['target', 'diagnosis'], axis=1)
    y = df['target']
    return X, y

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and return results"""
    # Scale features
    scaler = CustomStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_test_scaled = scaler.transform(X_test.values)
    
    # Convert to numpy arrays
    y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
    y_test_np = y_test.values if hasattr(y_test, 'values') else y_test
    
    # Define models
    models = {
        'Logistic Regression': CustomLogisticRegression(max_iter=1000),
        'K-Nearest Neighbors': CustomKNN(k=5),
        'Naive Bayes': CustomNaiveBayes()
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        try:
            # Train model
            model.fit(X_train_scaled, y_train_np)
            trained_models[name] = model
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Get probabilities (handle different model outputs)
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test_scaled)
                # Handle different probability array shapes
                if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                    y_pred_proba = y_pred_proba[:, 1]
                else:
                    y_pred_proba = y_pred_proba.flatten()
            else:
                y_pred_proba = np.zeros_like(y_pred, dtype=float)
            
            # Calculate metrics
            results[name] = {
                'Accuracy': accuracy_score(y_test_np, y_pred),
                'Precision': precision_score(y_test_np, y_pred),
                'Recall': recall_score(y_test_np, y_pred),
                'F1-Score': f1_score(y_test_np, y_pred),
                'ROC-AUC': roc_auc_score(y_test_np, y_pred_proba) if len(np.unique(y_pred_proba)) > 1 else 0.5,
                'Predictions': y_pred,
                'Probabilities': y_pred_proba,
                'Model': model
            }
        except Exception as e:
            st.error(f"Error training {name}: {str(e)}")
            continue
    
    return results, scaler, X_train_scaled, X_test_scaled

def plot_correlation_matrix(df):
    """Create correlation matrix heatmap"""
    # Select numerical features (exclude target and diagnosis)
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features = [col for col in numerical_features if col not in ['target']]
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_features].corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(15, 12))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": .5})
    ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_feature_distributions(df):
    """Plot feature distributions by diagnosis"""
    # Select important features for visualization
    important_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                         'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points']
    
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=important_features,
        vertical_spacing=0.08,
        horizontal_spacing=0.08
    )
    
    colors = ['#ff6b6b', '#4ecdc4']
    
    for i, feature in enumerate(important_features):
        row = i // 4 + 1
        col = i % 4 + 1
        
        for j, diagnosis in enumerate(['Malignant', 'Benign']):
            data = df[df['diagnosis'] == diagnosis][feature]
            
            fig.add_trace(
                go.Histogram(
                    x=data,
                    name=diagnosis,
                    marker_color=colors[j],
                    opacity=0.7,
                    legendgroup=diagnosis,
                    showlegend=(i == 0)
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        title_text="Feature Distributions by Diagnosis",
        title_x=0.5,
        height=600,
        barmode='overlay'
    )
    
    return fig

def create_model_comparison_chart(results):
    """Create model comparison chart"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    models = list(results.keys())
    
    fig = go.Figure()
    
    for metric in metrics:
        values = [results[model][metric] for model in models]
        fig.add_trace(go.Scatter(
            x=models,
            y=values,
            mode='lines+markers',
            name=metric,
            line=dict(width=3),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Models',
        yaxis_title='Score',
        yaxis=dict(range=[0, 1]),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def plot_confusion_matrices(results, y_test):
    """Plot confusion matrices for all models"""
    n_models = len(results)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, (name, result) in enumerate(results.items()):
        row = idx // cols
        col = idx % cols
        
        cm = confusion_matrix(y_test, result['Predictions'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   ax=axes[row, col],
                   xticklabels=['Malignant', 'Benign'],
                   yticklabels=['Malignant', 'Benign'])
        axes[row, col].set_title(f'{name}')
        axes[row, col].set_xlabel('Predicted')
        axes[row, col].set_ylabel('Actual')
    
    # Hide empty subplots
    for idx in range(n_models, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig

def plot_roc_curves(results, y_test):
    """Plot ROC curves for all models"""
    fig = go.Figure()
    
    for name, result in results.items():
        if result['Probabilities'] is not None and len(np.unique(result['Probabilities'])) > 1:
            fpr, tpr, _ = roc_curve(y_test, result['Probabilities'])
            auc_score = auc(fpr, tpr)
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{name} (AUC = {auc_score:.3f})',
                line=dict(width=2)
            ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        width=800, height=600
    )
    
    return fig

def main():
    st.markdown('<div class="main-header">🏥 Breast Cancer Detection System</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
        <p style="margin: 0; text-align: center; font-size: 1.1rem; color: #2c3e50;">
            This application uses machine learning to classify breast cancer tumors as malignant or benign 
            based on various tumor characteristics from the Wisconsin Breast Cancer Dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df, data = load_data()
    X, y = prepare_features(df)
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Select Page", [
        "📊 Dataset Overview", 
        "🔍 Exploratory Data Analysis", 
        "🤖 Model Training & Evaluation",
        "🎯 Make Predictions"
    ])
    
    if page == "📊 Dataset Overview":
        st.markdown('<div class="sub-header">Dataset Information</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Total Samples</h3>
                <h2 style="color: #e91e63;">{}</h2>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>Features</h3>
                <h2 style="color: #e91e63;">{}</h2>
            </div>
            """.format(len(df.columns) - 2), unsafe_allow_html=True)
        
        with col3:
            benign_count = (df['target'] == 1).sum()
            st.markdown("""
            <div class="metric-card">
                <h3>Benign Cases</h3>
                <h2 style="color: #28a745;">{}</h2>
            </div>
            """.format(benign_count), unsafe_allow_html=True)
        
        with col4:
            malignant_count = (df['target'] == 0).sum()
            st.markdown("""
            <div class="metric-card">
                <h3>Malignant Cases</h3>
                <h2 style="color: #dc3545;">{}</h2>
            </div>
            """.format(malignant_count), unsafe_allow_html=True)
        
        st.markdown('<div class="sub-header">Dataset Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
        
        # Dataset description
        st.markdown('<div class="sub-header">Feature Descriptions</div>', unsafe_allow_html=True)
        st.markdown(data.DESCR)
        
        # Target distribution
        fig = px.pie(df, names='diagnosis', title='Distribution of Diagnoses',
                    color_discrete_map={'Benign': '#28a745', 'Malignant': '#dc3545'})
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🔍 Exploratory Data Analysis":
        st.markdown('<div class="sub-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
        
        # Feature distributions
        st.plotly_chart(plot_feature_distributions(df), use_container_width=True)
        
        # Correlation matrix
        st.markdown('<div class="sub-header">Feature Correlation Matrix</div>', unsafe_allow_html=True)
        fig = plot_correlation_matrix(df)
        st.pyplot(fig)
        
        # Statistical summary
        st.markdown('<div class="sub-header">Statistical Summary</div>', unsafe_allow_html=True)
        st.dataframe(df.describe(), use_container_width=True)
    
    elif page == "🤖 Model Training & Evaluation":
        st.markdown('<div class="sub-header">Model Training & Evaluation</div>', unsafe_allow_html=True)
        
        # Split data using custom implementation
        X_train, X_test, y_train, y_test = custom_train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        with st.spinner("Training models..."):
            results, scaler, X_train_scaled, X_test_scaled = train_models(
                X_train, X_test, y_train, y_test
            )
        
        if not results:
            st.error("No models were successfully trained. Please check the error messages above.")
            return
        
        # Model comparison
        st.plotly_chart(create_model_comparison_chart(results), use_container_width=True)
        
        # Performance metrics table
        st.markdown('<div class="sub-header">Performance Metrics</div>', unsafe_allow_html=True)
        metrics_df = pd.DataFrame(results).T[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]
        metrics_df = metrics_df.round(4)
        st.dataframe(metrics_df, use_container_width=True)
        
        # Best model highlight
        best_model_name = metrics_df['Accuracy'].idxmax()
        best_accuracy = metrics_df.loc[best_model_name, 'Accuracy']
        
        st.success(f"🏆 Best Model: **{best_model_name}** with {best_accuracy:.4f} accuracy")
        
        # Confusion matrices
        st.markdown('<div class="sub-header">Confusion Matrices</div>', unsafe_allow_html=True)
        y_test_np = y_test.values if hasattr(y_test, 'values') else y_test
        fig = plot_confusion_matrices(results, y_test_np)
        st.pyplot(fig)
        
        # ROC curves
        st.markdown('<div class="sub-header">ROC Curves</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_roc_curves(results, y_test_np), use_container_width=True)
        
        # Store best model in session state
        st.session_state['best_model'] = results[best_model_name]['Model']
        st.session_state['scaler'] = scaler
        st.session_state['feature_names'] = X.columns.tolist()
    
    elif page == "🎯 Make Predictions":
        st.markdown('<div class="sub-header">Make Predictions</div>', unsafe_allow_html=True)
        
        if 'best_model' not in st.session_state:
            st.warning("⚠️ Please train models first by visiting the 'Model Training & Evaluation' page.")
            return
        
        st.info("Enter the tumor characteristics below to get a prediction:")
        
        # Create input form
        feature_names = st.session_state['feature_names']
        
        # Group features by type for better organization
        mean_features = [f for f in feature_names if 'mean' in f]
        se_features = [f for f in feature_names if 'error' in f]
        worst_features = [f for f in feature_names if 'worst' in f]
        
        # Default values (approximate means from the dataset)
        default_values = {
            'mean radius': 14.0, 'mean texture': 19.0, 'mean perimeter': 92.0,
            'mean area': 655.0, 'mean smoothness': 0.096, 'mean compactness': 0.104,
            'mean concavity': 0.089, 'mean concave points': 0.048, 'mean symmetry': 0.181,
            'mean fractal dimension': 0.063, 'radius error': 0.405, 'texture error': 1.217,
            'perimeter error': 2.866, 'area error': 40.337, 'smoothness error': 0.007,
            'compactness error': 0.025, 'concavity error': 0.032, 'concave points error': 0.012,
            'symmetry error': 0.021, 'fractal dimension error': 0.004, 'worst radius': 16.269,
            'worst texture': 25.677, 'worst perimeter': 107.261, 'worst area': 880.583,
            'worst smoothness': 0.132, 'worst compactness': 0.254, 'worst concavity': 0.272,
            'worst concave points': 0.115, 'worst symmetry': 0.290, 'worst fractal dimension': 0.084
        }
        
        # Input method selection
        input_method = st.radio("Choose input method:", 
                               ["Manual Input", "Random Sample from Dataset"])
        
        input_values = {}
        
        if input_method == "Manual Input":
            # Organize inputs in tabs
            tab1, tab2, tab3 = st.tabs(["Mean Features", "Standard Error Features", "Worst Features"])
            
            with tab1:
                st.markdown("**Mean Features**")
                cols = st.columns(3)
                for i, feature in enumerate(mean_features):
                    with cols[i % 3]:
                        input_values[feature] = st.number_input(
                            feature.replace('mean ', '').title(),
                            value=default_values.get(feature, 0.0),
                            format="%.6f",
                            key=f"input_{feature}"
                        )
            
            with tab2:
                st.markdown("**Standard Error Features**")
                cols = st.columns(3)
                for i, feature in enumerate(se_features):
                    with cols[i % 3]:
                        input_values[feature] = st.number_input(
                            feature.replace(' error', ' SE').title(),
                            value=default_values.get(feature, 0.0),
                            format="%.6f",
                            key=f"input_{feature}"
                        )
            
            with tab3:
                st.markdown("**Worst Features**")
                cols = st.columns(3)
                for i, feature in enumerate(worst_features):
                    with cols[i % 3]:
                        input_values[feature] = st.number_input(
                            feature.replace('worst ', '').title() + ' (Worst)',
                            value=default_values.get(feature, 0.0),
                            format="%.6f",
                            key=f"input_{feature}"
                        )
        
        else:  # Random sample
            if st.button("Generate Random Sample"):
                sample_idx = np.random.randint(0, len(df))
                sample_data = df.iloc[sample_idx]
                st.session_state['sample_data'] = sample_data
            
            if 'sample_data' in st.session_state:
                sample_data = st.session_state['sample_data']
                st.write(f"**Sample from dataset (Index: {sample_data.name})**")
                st.write(f"**Actual Diagnosis: {sample_data['diagnosis']}**")
                
                input_values = {feature: sample_data[feature] for feature in feature_names}
                
                # Display the sample values
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("**Mean Features:**")
                    for feature in mean_features[:len(mean_features)//3 + 1]:
                        st.write(f"{feature}: {input_values[feature]:.6f}")
                
                with col2:
                    st.write("**Standard Error Features:**")
                    for feature in se_features[:len(se_features)//3 + 1]:
                        st.write(f"{feature}: {input_values[feature]:.6f}")
                
                with col3:
                    st.write("**Worst Features:**")
                    for feature in worst_features[:len(worst_features)//3 + 1]:
                        st.write(f"{feature}: {input_values[feature]:.6f}")
            else:
                st.info("Click 'Generate Random Sample' to get a random sample from the dataset.")
                return
        
        # Make prediction
        if st.button("🔮 Make Prediction", type="primary"):
            try:
                # Prepare input data
                input_array = np.array([input_values[feature] for feature in feature_names]).reshape(1, -1)
                
                # Scale input data
                input_scaled = st.session_state['scaler'].transform(input_array)
                
                # Make prediction
                prediction = st.session_state['best_model'].predict(input_scaled)[0]
                
                # Get probabilities
                if hasattr(st.session_state['best_model'], 'predict_proba'):
                    prediction_proba = st.session_state['best_model'].predict_proba(input_scaled)
                    # Handle different probability array shapes
                    if len(prediction_proba.shape) > 1 and prediction_proba.shape[1] > 1:
                        malignant_prob = prediction_proba[0, 0]
                        benign_prob = prediction_proba[0, 1]
                    else:
                        malignant_prob = prediction_proba[0]
                        benign_prob = 1 - prediction_proba[0]
                else:
                    # For models without probability estimation
                    malignant_prob = 1.0 if prediction == 0 else 0.0
                    benign_prob = 1.0 if prediction == 1 else 0.0
                
                # Display results
                if prediction == 1:  # Benign
                    st.markdown(f"""
                    <div class="prediction-result benign">
                        ✅ BENIGN<br>
                        Confidence: {benign_prob:.2%}
                    </div>
                    """, unsafe_allow_html=True)
                    st.success("The tumor is predicted to be **BENIGN** (non-cancerous).")
                else:  # Malignant
                    st.markdown(f"""
                    <div class="prediction-result malignant">
                        ⚠️ MALIGNANT<br>
                        Confidence: {malignant_prob:.2%}
                    </div>
                    """, unsafe_allow_html=True)
                    st.error("The tumor is predicted to be **MALIGNANT** (cancerous).")
                
                # Display probability distribution
                prob_df = pd.DataFrame({
                    'Diagnosis': ['Malignant', 'Benign'],
                    'Probability': [malignant_prob, benign_prob]
                })
                
                fig = px.bar(prob_df, x='Diagnosis', y='Probability', 
                            color='Diagnosis', color_discrete_map={
                                'Malignant': '#dc3545', 
                                'Benign': '#28a745'
                            })
                fig.update_layout(title='Prediction Confidence', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()
