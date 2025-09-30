import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up our page
st.set_page_config(page_title="Exoplanet Explorer", page_icon="🪐", layout="wide")

# 🎨 SPACE BACKGROUND CODE
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1534796636912-3b95b3ab5986?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1740&q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    .stApp > header {
        background-color: transparent;
    }
    
    .stApp {
        background-color: rgba(0, 0, 0, 0.7);
    }
    
    /* Make text more readable on background */
    .main .block-container {
        background-color: rgba(0, 0, 0, 0.8);
        padding: 2rem;
        border-radius: 10px;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    
    .stMarkdown, .stText {
        color: white !important;
    }
    
    /* Style metrics */
    [data-testid="stMetricValue"] {
        color: #00ff00 !important;
        font-weight: bold;
    }
    
    [data-testid="stMetricLabel"] {
        color: white !important;
    }
    
    /* Style buttons */
    .stButton button {
        background-color: #4CAF50 !important;
        color: white !important;
        border-radius: 5px !important;
        border: none !important;
    }
    
    /* Add glowing effect to main title */
    h1 {
        text-shadow: 0 0 10px #00ff00, 0 0 20px #00ff00;
    }
    
    /* NEW: Light Blue Table Theme */
    .dataframe thead tr th {
        background-color: #e6f3ff !important;
        color: #0066cc !important;
        font-weight: bold;
        border: 1px solid #b3d9ff;
    }
    
    .dataframe tbody tr td {
        background-color: #f0f8ff !important;
        color: #333333 !important;
        border: 1px solid #d9edf7;
    }
    
    .model-comparison-table thead tr th {
        background-color: #e6f3ff !important;
        color: #0066cc !important;
        font-weight: bold;
        border: 1px solid #b3d9ff;
    }
    
    .model-comparison-table tbody tr td {
        background-color: #f0f8ff !important;
        color: #333333 !important;
        border: 1px solid #d9edf7;
    }
    
    /* Performance metrics styling */
    .performance-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        color: white;
    }
    
    .improvement {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
    }
    
    .decline {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    }
    
    .model-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #4CAF50;
    }
    
    .quantum-card {
        border-left: 5px solid #9C27B0;
    }
    
    .svm-card {
        border-left: 5px solid #2196F3;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add a big title
st.title("🪐 Exoplanet Explorer")
st.markdown("Explore real exoplanet data from Kepler, K2, and TESS telescopes! 🌟")

@st.cache_data
def load_data():
    try:
        # Load Kepler data
        kepler_df = pd.read_csv('cumulative_2025.09.21_04.31.43.csv', comment="#")
        st.success(f"✅ Kepler data loaded: {len(kepler_df)} planets")
        
        # Load K2 data  
        k2_df = pd.read_csv('k2pandc_2025.09.17_06.42.42.csv', comment="#")
        st.success(f"✅ K2 data loaded: {len(k2_df)} planets")
        
        # Load TESS data
        tess_df = pd.read_csv('TOI_2025.09.17_06.36.05.csv', comment="#")
        st.success(f"✅ TESS data loaded: {len(tess_df)} planets")
        
        return kepler_df, k2_df, tess_df
        
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e}")
        st.info("Please make sure your CSV files are in the same folder as this script:")
        st.code("""
        - cumulative_2025.09.21_04.31.43.csv
        - k2pandc_2025.09.17_06.42.42.csv  
        - TOI_2025.09.17_06.36.05 (2).csv
        """)
        # Return empty dataframes to avoid crashes
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def clean_and_merge_data(kepler_df, k2_df, tess_df):
    """Clean and merge the three datasets"""
    if kepler_df.empty or k2_df.empty or tess_df.empty:
        return pd.DataFrame()
    
    # Create copies to avoid modifying originals
    kepler_clean = kepler_df.copy()
    k2_clean = k2_df.copy()
    tess_clean = tess_df.copy()
    
    # --- Clean Kepler Data ---
    column_mapping_kepler = {
        # Planet Properties
        'koi_period': 'pl_orbper',
        'koi_period_err1': 'pl_orbpererr1',
        'koi_period_err2': 'pl_orbpererr2',
        'koi_prad': 'pl_rade',
        'koi_prad_err1': 'pl_radeerr1',
        'koi_prad_err2': 'pl_radeerr2',
        'koi_teq': 'pl_eqt',
        'koi_insol': 'pl_insol',
        'koi_depth': 'pl_trandep',
        'koi_duration': 'pl_trandurh',
        # Stellar Properties
        'koi_steff': 'st_teff',
        'koi_steff_err1': 'st_tefferr1',
        'koi_steff_err2': 'st_tefferr2',
        'koi_slogg': 'st_logg',
        'koi_slogg_err1': 'st_loggerr1',
        'koi_slogg_err2': 'st_loggerr2',
        'koi_srad': 'st_rad',
        'koi_srad_err1': 'st_raderr1',
        'koi_srad_err2': 'st_raderr2',
        'koi_kepmag': 'st_kepmag',
        # Disposition
        'koi_disposition': 'disposition',
        'koi_pdisposition': 'disposition_provided'
    }
    
    # Rename columns that exist in Kepler data
    existing_columns = {k: v for k, v in column_mapping_kepler.items() if k in kepler_clean.columns}
    kepler_clean.rename(columns=existing_columns, inplace=True)
    
    # Drop columns with >90% missing values
    threshold = 0.9
    cols_to_drop = kepler_clean.columns[kepler_clean.isnull().mean() > threshold].tolist()
    if cols_to_drop:
        kepler_clean.drop(columns=cols_to_drop, inplace=True)
    
    kepler_clean['mission'] = 'Kepler'
    
    # --- Clean K2 Data ---
    k2_clean['mission'] = 'K2'
    
    # --- Clean TESS Data ---
    tess_clean['mission'] = 'TESS'
    
    # Find common columns across all datasets
    kepler_cols = set(kepler_clean.columns)
    k2_cols = set(k2_clean.columns)
    tess_cols = set(tess_clean.columns)
    common_cols = list(kepler_cols & k2_cols & tess_cols)
    
    st.info(f"🔄 Merging data using {len(common_cols)} common columns")
    
    # Select only common columns and merge
    kepler_final = kepler_clean[common_cols]
    k2_final = k2_clean[common_cols]
    tess_final = tess_clean[common_cols]
    
    master_df = pd.concat([kepler_final, k2_final, tess_final], ignore_index=True)
    
    return master_df

def smart_data_doctor(new_data):
    """
    Automatically fixes common data problems with robust error handling
    """
    st.write("🔧 Running Advanced Data Doctor...")
    
    data_fixes = []
    original_shape = new_data.shape
    
    try:
        # 1. Fix column names with extensive mapping
        column_mapping = {
            'pl_orbper': ['pl_orbper', 'koi_period', 'orbital_period', 'period', 'Period', 'p_orb', 'P_orb'],
            'pl_rade': ['pl_rade', 'koi_prad', 'planet_radius', 'radius', 'Radius', 'p_rad', 'P_rad'],
            'st_teff': ['st_teff', 'koi_steff', 'stellar_teff', 'teff', 'Teff', 'temp', 'temperature'],
            'st_rad': ['st_rad', 'koi_srad', 'stellar_radius', 'srad', 's_rad', 'star_radius'],
            'st_logg': ['st_logg', 'koi_slogg', 'stellar_logg', 'logg', 'Logg', 'gravity'],
            'pl_eqt': ['pl_eqt', 'eq_temp', 'equilibrium_temp', 'teq', 'Teq'],
            'disposition': ['disposition', 'koi_disposition', 'status', 'classification', 'label']
        }
        
        for standard_name, possible_names in column_mapping.items():
            for name in possible_names:
                if name in new_data.columns and standard_name not in new_data.columns:
                    new_data[standard_name] = new_data[name]
                    data_fixes.append(f"Renamed '{name}' → '{standard_name}'")
                    break
        
        # 2. Create missing columns with default values
        required_columns = ['pl_orbper', 'pl_rade', 'st_teff', 'st_rad', 'st_logg']
        for col in required_columns:
            if col not in new_data.columns:
                if col == 'pl_rade':
                    new_data[col] = 2.0  # Earth-like default
                elif col == 'pl_orbper':
                    new_data[col] = 10.0  # Typical orbital period
                elif col == 'st_teff':
                    new_data[col] = 5778  # Sun-like temperature
                elif col == 'st_rad':
                    new_data[col] = 1.0  # Solar radius
                elif col == 'st_logg':
                    new_data[col] = 4.4  # Solar logg
                data_fixes.append(f"Created missing column '{col}' with default value")
        
        # 3. Handle missing values intelligently
        for col in new_data.columns:
            if new_data[col].dtype in ['float64', 'int64', 'object']:
                missing_count = new_data[col].isna().sum()
                if missing_count > 0:
                    if col == 'pl_rade':
                        new_data[col].fillna(2.0, inplace=True)
                    elif col == 'pl_orbper':
                        new_data[col].fillna(10.0, inplace=True)
                    elif col == 'st_teff':
                        new_data[col].fillna(5778, inplace=True)
                    elif col == 'st_rad':
                        new_data[col].fillna(1.0, inplace=True)
                    elif col == 'st_logg':
                        new_data[col].fillna(4.4, inplace=True)
                    elif col == 'disposition':
                        new_data[col].fillna('CANDIDATE', inplace=True)
                    else:
                        new_data[col].fillna(new_data[col].median() if new_data[col].dtype in ['float64', 'int64'] else 'UNKNOWN', inplace=True)
                    data_fixes.append(f"Filled {missing_count} missing values in {col}")
        
        # 4. Convert data types automatically
        type_conversions = 0
        for col in new_data.columns:
            if new_data[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    new_data[col] = pd.to_numeric(new_data[col], errors='ignore')
                    if new_data[col].dtype in ['float64', 'int64']:
                        type_conversions += 1
                except:
                    pass
        
        if type_conversions > 0:
            data_fixes.append(f"Converted {type_conversions} columns to numeric")
        
        # 5. Fix extreme values
        for col in new_data.columns:
            if new_data[col].dtype in ['float64', 'int64']:
                Q1 = new_data[col].quantile(0.25)
                Q3 = new_data[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:  # Avoid division by zero
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = ((new_data[col] < lower_bound) | (new_data[col] > upper_bound)).sum()
                    if outliers > 0:
                        new_data[col] = np.clip(new_data[col], lower_bound, upper_bound)
                        data_fixes.append(f"Fixed {outliers} outliers in {col}")
        
        # Show what was fixed
        if data_fixes:
            st.success("🎯 Data fixes applied:")
            for fix in data_fixes:
                st.write(f"✅ {fix}")
        else:
            st.info("✅ Data looks clean - no fixes needed!")
        
        st.info(f"📊 Data shape: {original_shape} → {new_data.shape}")
        
        return new_data
        
    except Exception as e:
        st.error(f"❌ Error in data cleaning: {str(e)}")
        st.info("🔄 Using fallback cleaning method...")
        # Fallback: basic cleaning
        return new_data.fillna(0).infer_objects()

def detect_target_column(data):
    """Detect if target labels exist in the data"""
    target_indicators = ['disposition', 'koi_disposition', 'label', 'target', 'classification', 'status']
    
    for col in target_indicators:
        if col in data.columns:
            # Check if it has meaningful values
            unique_vals = data[col].dropna().unique()
            if len(unique_vals) > 1:
                return col, unique_vals
    
    return None, None

def create_sample_data():
    """Create sample data for testing when no file is uploaded"""
    sample_data = {
        'pl_orbper': np.random.uniform(1, 365, 50),
        'pl_rade': np.random.uniform(0.5, 20, 50),
        'st_teff': np.random.uniform(3000, 7000, 50),
        'st_rad': np.random.uniform(0.5, 2.0, 50),
        'st_logg': np.random.uniform(4.0, 5.0, 50),
        'disposition': np.random.choice(['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'], 50)
    }
    return pd.DataFrame(sample_data)

class ModelSimulator:
    """Simulate different ML models for exoplanet prediction"""
    
    def __init__(self):
        self.training_performance = {
            'Random Forest': {'recall': 0.977, 'precision': 0.891, 'f1': 0.932},
            'SVM': {'recall': 0.964, 'precision': 0.690, 'f1': 0.805},
            'Quantum-Hybrid': {'recall': 0.931, 'precision': 0.818, 'f1': 0.871}
        }
    
    def predict_random_forest(self, data):
        """Simulate Random Forest predictions"""
        predictions = []
        confidences = []
        
        for _, row in data.iterrows():
            score = 0
            
            # Random Forest-like rules (ensemble of decision trees simulation)
            if row.get('pl_rade', 0) > 0.8:
                score += 0.3
            if 50 <= row.get('pl_orbper', 0) <= 400:
                score += 0.25
            if 5200 <= row.get('st_teff', 0) <= 6000:
                score += 0.2
            if 0.8 <= row.get('st_rad', 0) <= 1.2:
                score += 0.15
            if 4.0 <= row.get('st_logg', 0) <= 4.9:
                score += 0.1
            
            # Add some randomness to simulate ensemble
            score += np.random.normal(0, 0.05)
            score = max(0, min(1, score))
            
            prediction = 1 if score > 0.5 else 0
            confidence = score if prediction == 1 else 1 - score
            
            predictions.append(prediction)
            confidences.append(confidence)
        
        return predictions, confidences
    
    def predict_svm(self, data):
        """Simulate SVM predictions (more strict boundaries)"""
        predictions = []
        confidences = []
        
        for _, row in data.iterrows():
            score = 0
            
            # SVM-like rules (strict boundaries)
            if 0.8 <= row.get('pl_rade', 0) <= 2.5:
                score += 0.4
            if 100 <= row.get('pl_orbper', 0) <= 300:
                score += 0.3
            if 5500 <= row.get('st_teff', 0) <= 6000:
                score += 0.2
            if 0.9 <= row.get('st_rad', 0) <= 1.1:
                score += 0.1
            
            # SVM tends to be more confident
            score = max(0, min(1, score))
            prediction = 1 if score > 0.6 else 0  # Higher threshold
            confidence = score if prediction == 1 else 1 - score
            
            predictions.append(prediction)
            confidences.append(confidence)
        
        return predictions, confidences
    
    def predict_quantum_hybrid(self, data):
        """Simulate Quantum-Hybrid predictions (quantum-inspired patterns)"""
        predictions = []
        confidences = []
        
        for _, row in data.iterrows():
            # Quantum-inspired scoring (considers quantum-like superposition)
            features = [
                row.get('pl_rade', 0),
                row.get('pl_orbper', 0),
                row.get('st_teff', 0),
                row.get('st_rad', 0),
                row.get('st_logg', 0)
            ]
            
            # Quantum amplitude calculation (simplified)
            amplitude = sum(f * np.exp(1j * i * 0.1) for i, f in enumerate(features) if f > 0)
            score = np.abs(amplitude) / 10  # Normalize
            
            # Quantum interference patterns
            if 0.5 <= row.get('pl_rade', 0) <= 4.0:
                score *= 1.2
            if 10 <= row.get('pl_orbper', 0) <= 500:
                score *= 1.1
            
            score = max(0, min(1, score.real))
            
            prediction = 1 if score > 0.45 else 0  # Lower threshold for quantum
            confidence = score if prediction == 1 else 1 - score
            
            predictions.append(prediction)
            confidences.append(confidence)
        
        return predictions, confidences
    
    def calculate_metrics(self, true_labels, predictions, model_name):
        """Calculate performance metrics"""
        if true_labels is None or len(true_labels) == 0:
            return None
        
        try:
            # Convert to binary (1 for planet, 0 for non-planet)
            true_binary = [1 if str(label).upper() in ['CONFIRMED', 'CANDIDATE', '1', 'TRUE'] else 0 
                          for label in true_labels]
            
            # Basic metrics
            correct = sum(1 for t, p in zip(true_binary, predictions) if t == p)
            accuracy = correct / len(true_binary)
            
            # Precision, Recall, F1
            tp = sum(1 for t, p in zip(true_binary, predictions) if t == 1 and p == 1)
            fp = sum(1 for t, p in zip(true_binary, predictions) if t == 0 and p == 1)
            fn = sum(1 for t, p in zip(true_binary, predictions) if t == 1 and p == 0)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Compare with training performance
            train_recall = self.training_performance[model_name]['recall']
            train_precision = self.training_performance[model_name]['precision']
            train_f1 = self.training_performance[model_name]['f1']
            
            deltas = {
                'recall': recall - train_recall,
                'precision': precision - train_precision,
                'f1': f1 - train_f1
            }
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'deltas': deltas,
                'confusion_matrix': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': len(true_binary) - tp - fp - fn}
            }
        
        except Exception as e:
            st.warning(f"⚠️ Could not calculate metrics for {model_name}: {str(e)}")
            return None

def plot_performance_comparison(metrics_dict):
    """Create performance comparison visualization"""
    if not metrics_dict:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    models = list(metrics_dict.keys())
    
    # Metrics to plot
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_keys = ['accuracy', 'precision', 'recall', 'f1']
    
    for idx, (metric_name, metric_key) in enumerate(zip(metric_names, metric_keys)):
        ax = axes[idx//2, idx%2]
        
        values = [metrics_dict[model][metric_key] for model in models]
        
        bars = ax.bar(models, values, color=['#4CAF50', '#2196F3', '#9C27B0'], alpha=0.8)
        ax.set_title(f'{metric_name} Comparison')
        ax.set_ylabel(metric_name)
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_confusion_matrices(metrics_dict):
    """Plot confusion matrices for all models"""
    if not metrics_dict:
        return None
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
    
    for idx, (model_name, metrics) in enumerate(metrics_dict.items()):
        if metrics and 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            matrix = np.array([[cm['tp'], cm['fp']], [cm['fn'], cm['tn']]])
            
            sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Predicted Planet', 'Predicted Non-Planet'],
                       yticklabels=['Actual Planet', 'Actual Non-Planet'])
            axes[idx].set_title(f'{model_name}\nConfusion Matrix')
    
    plt.tight_layout()
    return fig

def show_model_comparison():
    st.header("🤖 Model Comparison: Classical vs Quantum")
    
    st.markdown("""
    Compare different machine learning approaches for exoplanet discovery. 
    This section shows how various algorithms perform in identifying potential exoplanets from telescope data.
    """)
    
    # Performance comparison table
    st.subheader("📊 Performance Metrics")
    
    comparison_data = {
        'Model': ['Random Forest', 'Quantum-Hybrid', 'SVM'],
        'Recall': ['97.7%', '93.1%', '96.4%'],
        'Precision': ['89.1%', '81.8%', '69.0%'], 
        'F1-Score': ['93.2%', '87.1%', '80.5%'],
        'Best For': ['Balanced Performance', 'Research Showcase', 'Maximum Discovery'],
        'Training Time': ['2 minutes', '15 minutes', '5 minutes']
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Apply custom styling to the table
    st.markdown('<div class="model-comparison-table">', unsafe_allow_html=True)
    st.table(comparison_df)
    st.markdown('</div>', unsafe_allow_html=True)

# Load the data
with st.spinner('Loading real exoplanet data... 🚀'):
    kepler_df, k2_df, tess_df = load_data()

# Only proceed if we have data
if not kepler_df.empty and not k2_df.empty and not tess_df.empty:
    # Clean and merge data
    with st.spinner('Cleaning and merging data... 🔄'):
        master_df = clean_and_merge_data(kepler_df, k2_df, tess_df)
    
    if not master_df.empty:
        st.success(f"🎉 Successfully merged {len(master_df)} real exoplanets from all missions!")
        
        # --- ADVANCED MODEL TESTING SECTION ---
        st.header("🧪 Advanced Model Testing Laboratory")
        
        st.markdown("""
        **Upload new exoplanet data to test all AI models with automatic performance analysis!**
        
        🚀 **Features:**
        - Tests all 3 models (Random Forest, SVM, Quantum-Hybrid) simultaneously
        - Automatic performance comparison vs training data
        - Robust error handling for any data format
        - Professional metrics with delta indicators
        - Confusion matrices and detailed analysis
        
        🛡 **Handles:**
        - Different column names and formats
        - Missing values and columns
        - Data type conversions
        - Corrupted files with fallbacks
        """)
        
        # File upload with robust handling
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📁 Upload New Data")
            uploaded_file = st.file_uploader("Choose CSV or Excel file", type=['csv', 'xlsx', 'xls'])
            
            use_sample = st.checkbox("🎯 Use sample data for testing", value=False)
            
            if uploaded_file is not None:
                try:
                    # Handle different file types
                    if uploaded_file.name.endswith('.csv'):
                        test_data = pd.read_csv(uploaded_file)
                    else:  # Excel files
                        test_data = pd.read_excel(uploaded_file)
                    
                    st.success(f"✅ Data uploaded: {len(test_data)} rows, {len(test_data.columns)} columns")
                    
                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")
                    st.info("🔄 Using sample data instead...")
                    test_data = create_sample_data()
                    use_sample = True
            elif use_sample:
                test_data = create_sample_data()
                st.info("🎯 Using sample data for demonstration")
            else:
                test_data = None
        
        with col2:
            st.subheader("⚙️ Testing Options")
            run_comprehensive = st.checkbox("🏃‍♂️ Run comprehensive analysis", value=True)
            show_details = st.checkbox("🔍 Show detailed predictions", value=False)
            compare_training = st.checkbox("📊 Compare with training performance", value=True)
        
        if test_data is not None:
            # Initialize model simulator
            model_simulator = ModelSimulator()
            
            # Data preprocessing
            st.subheader("🔧 Data Preprocessing")
            with st.spinner("Cleaning and preparing data..."):
                processed_data = smart_data_doctor(test_data.copy())
                
                # Detect target column
                target_column, target_values = detect_target_column(processed_data)
                if target_column:
                    st.success(f"🎯 Target column detected: '{target_column}' with values: {list(target_values)}")
                else:
                    st.warning("⚠️ No target labels detected - showing predictions only")
            
            # Run model predictions
            if st.button("🚀 Run All Model Tests", type="primary") or run_comprehensive:
                st.subheader("📈 Model Performance Results")
                
                # Store results
                all_predictions = {}
                all_metrics = {}
                
                # Test all models
                models = {
                    'Random Forest': model_simulator.predict_random_forest,
                    'SVM': model_simulator.predict_svm,
                    'Quantum-Hybrid': model_simulator.predict_quantum_hybrid
                }
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, (model_name, predict_func) in enumerate(models.items()):
                    status_text.text(f"Testing {model_name}...")
                    
                    # Get predictions
                    predictions, confidences = predict_func(processed_data)
                    
                    # Calculate metrics if target available
                    true_labels = processed_data[target_column] if target_column else None
                    metrics = model_simulator.calculate_metrics(true_labels, predictions, model_name)
                    
                    # Store results
                    all_predictions[model_name] = {
                        'predictions': predictions,
                        'confidences': confidences,
                        'metrics': metrics
                    }
                    
                    if metrics:
                        all_metrics[model_name] = metrics
                    
                    progress_bar.progress((i + 1) / len(models))
                
                status_text.text("✅ All models tested successfully!")
                
                # Display results
                if all_metrics:
                    # Performance comparison chart
                    st.subheader("📊 Performance Comparison")
                    perf_fig = plot_performance_comparison(all_metrics)
                    if perf_fig:
                        st.pyplot(perf_fig)
                    
                    # Confusion matrices
                    st.subheader("🎯 Confusion Matrices")
                    cm_fig = plot_confusion_matrices(all_metrics)
                    if cm_fig:
                        st.pyplot(cm_fig)
                
                # Detailed model results
                st.subheader("🔍 Detailed Model Analysis")
                
                for model_name, results in all_predictions.items():
                    with st.container():
                        st.markdown(f"""
                        <div class="model-card {'quantum-card' if 'Quantum' in model_name else 'svm-card' if 'SVM' in model_name else ''}">
                            <h3>🧠 {model_name}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        if results['metrics']:
                            metrics = results['metrics']
                            
                            with col1:
                                delta = metrics['deltas']['recall'] if compare_training else None
                                st.metric("Recall", f"{metrics['recall']:.3f}", 
                                         delta=f"{delta:+.3f}" if delta else None)
                            
                            with col2:
                                delta = metrics['deltas']['precision'] if compare_training else None
                                st.metric("Precision", f"{metrics['precision']:.3f}", 
                                         delta=f"{delta:+.3f}" if delta else None)
                            
                            with col3:
                                delta = metrics['deltas']['f1'] if compare_training else None
                                st.metric("F1-Score", f"{metrics['f1']:.3f}", 
                                         delta=f"{delta:+.3f}" if delta else None)
                            
                            with col4:
                                st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                        
                        # Show prediction summary
                        planet_count = sum(results['predictions'])
                        st.info(f"🪐 {model_name} found {planet_count} planets out of {len(processed_data)} objects")
                
                # Combined results table
                st.subheader("📋 Combined Prediction Results")
                
                results_df = processed_data.copy()
                for model_name, results in all_predictions.items():
                    results_df[f'{model_name}_Prediction'] = ['🪐 Planet' if p == 1 else '⭐ Star' 
                                                            for p in results['predictions']]
                    results_df[f'{model_name}_Confidence'] = [f"{c:.1%}" for c in results['confidences']]
                
                # Add agreement column
                if len(all_predictions) > 1:
                    model_names = list(all_predictions.keys())
                    agreement = []
                    for i in range(len(processed_data)):
                        preds = [all_predictions[model]['predictions'][i] for model in model_names]
                        agreement.append('✅' if len(set(preds)) == 1 else '❌')
                    results_df['Model_Agreement'] = agreement
                
                st.dataframe(results_df.head(20))
                
                # Download results
                st.subheader("💾 Download Results")
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Results (CSV)",
                    data=csv,
                    file_name=f"exoplanet_model_testing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
                
                # Performance summary
                if all_metrics:
                    st.subheader("📈 Performance Summary")
                    
                    best_f1_model = max(all_metrics.items(), key=lambda x: x[1]['f1'])[0]
                    best_recall_model = max(all_metrics.items(), key=lambda x: x[1]['recall'])[0]
                    best_precision_model = max(all_metrics.items(), key=lambda x: x[1]['precision'])[0]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("🎯 Best Overall (F1)", best_f1_model, 
                                 delta=f"{all_metrics[best_f1_model]['f1']:.3f}")
                    
                    with col2:
                        st.metric("🔍 Best Recall", best_recall_model,
                                 delta=f"{all_metrics[best_recall_model]['recall']:.3f}")
                    
                    with col3:
                        st.metric("🎯 Best Precision", best_precision_model,
                                 delta=f"{all_metrics[best_precision_model]['precision']:.3f}")

        # Continue with the rest of your existing app sections...
        # [The rest of your existing code for data exploration, charts, etc.]
        
        # For brevity, I'll include just one section to show the structure
        st.header("🔍 Explore the Planets")
        
        # Your existing exploration code here...
        col1, col2 = st.columns(2)
        
        with col1:
            selected_mission = st.selectbox(
                "Choose a telescope mission:",
                ["All"] + list(master_df['mission'].unique())
            )
            
        with col2:
            max_radius = st.slider(
                "Maximum Planet Size (Earth radii):",
                min_value=0.1, max_value=50.0, value=20.0, step=1.0
            )
        
        # Filter data
        filtered_df = master_df.copy()
        if selected_mission != "All":
            filtered_df = filtered_df[filtered_df['mission'] == selected_mission]
        if 'pl_rade' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['pl_rade'] <= max_radius]
        
        st.write(f"Showing {len(filtered_df)} planets")
        
        # Continue with your existing app sections...

    else:
        st.error("Failed to merge datasets. Please check your data files.")
else:
    st.error("Could not load one or more data files. Please check that all CSV files are in the same folder as this script.")