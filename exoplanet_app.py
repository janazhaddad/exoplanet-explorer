import streamlit as st
import subprocess
import sys

# Try to import required packages, install if missing
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    import joblib
    import os
    import base64
    from datetime import datetime
    import warnings
    warnings.filterwarnings('ignore')
    
except ImportError as e:
    st.error(f"❌ Missing required package: {e}")
    st.info("📦 Installing required packages... This may take a moment.")
    
    # Install missing packages with better error handling
    packages = [
        "pandas", 
        "numpy", 
        "matplotlib", 
        "seaborn", 
        "scikit-learn", 
        "joblib"
    ]
    
    successful_installs = []
    failed_installs = []
    
    for package in packages:
        try:
            # Use pip with timeout and better output handling
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package, "--user"
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                successful_installs.append(package)
                st.success(f"✅ Installed {package}")
            else:
                failed_installs.append(package)
                st.warning(f"⚠ Could not install {package}: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            failed_installs.append(package)
            st.warning(f"⚠ Timeout installing {package}")
        except Exception as ex:
            failed_installs.append(package)
            st.warning(f"⚠ Error installing {package}: {str(ex)}")
    
    if failed_installs:
        st.error(f"❌ Failed to install: {', '.join(failed_installs)}")
        st.info("🔄 Please try installing manually or refresh the page to retry.")
        st.stop()
    else:
        st.success("🎉 All packages installed successfully!")
        st.info("🔄 Refreshing the page...")
        st.experimental_rerun()

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
    
    /* Light Blue Table Theme */
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
st.markdown("🌟 Explore real exoplanet data from Kepler, K2, and TESS telescopes! 🌟")

@st.cache_data
def load_data():
    try:
        # Try to load data files, but provide sample data if files not found
        kepler_df, k2_df, tess_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        try:
            kepler_df = pd.read_csv('cumulative_2025.09.21_04.31.43.csv', comment="#")
        except FileNotFoundError:
            st.warning("Kepler data file not found. Using sample data.")
            kepler_df = create_sample_kepler_data()
        
        try:
            k2_df = pd.read_csv('k2pandc_2025.09.17_06.42.42.csv', comment="#")
        except FileNotFoundError:
            st.warning("K2 data file not found. Using sample data.")
            k2_df = create_sample_k2_data()
        
        try:
            tess_df = pd.read_csv('TOI_2025.09.17_06.36.05.csv', comment="#")
        except FileNotFoundError:
            st.warning("TESS data file not found. Using sample data.")
            tess_df = create_sample_tess_data()
        
        return kepler_df, k2_df, tess_df
        
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        # Return sample data as fallback
        return create_sample_kepler_data(), create_sample_k2_data(), create_sample_tess_data()

def create_sample_kepler_data():
    """Create realistic sample Kepler data"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'koi_period': np.random.exponential(50, n_samples),
        'koi_prad': np.random.lognormal(0.5, 0.8, n_samples),
        'koi_teq': np.random.normal(1500, 500, n_samples),
        'koi_insol': np.random.lognormal(0, 1, n_samples),
        'koi_steff': np.random.normal(5500, 500, n_samples),
        'koi_slogg': np.random.normal(4.4, 0.2, n_samples),
        'koi_srad': np.random.normal(1.0, 0.2, n_samples),
        'koi_disposition': np.random.choice(['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'], n_samples, p=[0.3, 0.4, 0.3])
    }
    return pd.DataFrame(data)

def create_sample_k2_data():
    """Create realistic sample K2 data"""
    np.random.seed(43)
    n_samples = 500
    
    data = {
        'pl_orbper': np.random.exponential(30, n_samples),
        'pl_rade': np.random.lognormal(0.3, 0.7, n_samples),
        'pl_eqt': np.random.normal(1600, 600, n_samples),
        'pl_insol': np.random.lognormal(0.2, 0.9, n_samples),
        'st_teff': np.random.normal(5200, 600, n_samples),
        'st_logg': np.random.normal(4.5, 0.3, n_samples),
        'st_rad': np.random.normal(0.9, 0.3, n_samples),
        'disposition': np.random.choice(['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'], n_samples, p=[0.25, 0.45, 0.3])
    }
    return pd.DataFrame(data)

def create_sample_tess_data():
    """Create realistic sample TESS data"""
    np.random.seed(44)
    n_samples = 800
    
    data = {
        'pl_orbper': np.random.exponential(10, n_samples),
        'pl_rade': np.random.lognormal(0.2, 0.6, n_samples),
        'pl_eqt': np.random.normal(1400, 400, n_samples),
        'pl_insol': np.random.lognormal(-0.1, 0.8, n_samples),
        'st_teff': np.random.normal(5000, 700, n_samples),
        'st_logg': np.random.normal(4.6, 0.4, n_samples),
        'st_rad': np.random.normal(0.8, 0.4, n_samples),
        'disposition': np.random.choice(['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'], n_samples, p=[0.35, 0.4, 0.25])
    }
    return pd.DataFrame(data)

def clean_and_merge_data(kepler_df, k2_df, tess_df):
    """Clean and merge the three datasets"""
    if kepler_df.empty or k2_df.empty or tess_df.empty:
        st.warning("One or more datasets are empty. Using sample data.")
        return create_sample_combined_data()
    
    # Create copies to avoid modifying originals
    kepler_clean = kepler_df.copy()
    k2_clean = k2_df.copy()
    tess_clean = tess_df.copy()
    
    # --- Clean Kepler Data ---
    column_mapping_kepler = {
        'koi_period': 'pl_orbper',
        'koi_prad': 'pl_rade', 
        'koi_teq': 'pl_eqt',
        'koi_insol': 'pl_insol',
        'koi_steff': 'st_teff',
        'koi_slogg': 'st_logg',
        'koi_srad': 'st_rad',
        'koi_disposition': 'disposition'
    }
    
    # Rename columns that exist in Kepler data
    existing_columns = {k: v for k, v in column_mapping_kepler.items() if k in kepler_clean.columns}
    kepler_clean.rename(columns=existing_columns, inplace=True)
    
    # Fill missing values for common columns
    common_columns = ['pl_orbper', 'pl_rade', 'pl_eqt', 'st_teff', 'st_logg', 'st_rad', 'disposition']
    for col in common_columns:
        if col in kepler_clean.columns:
            if col == 'disposition':
                kepler_clean[col].fillna('UNKNOWN', inplace=True)
            else:
                kepler_clean[col].fillna(kepler_clean[col].median(), inplace=True)
    
    kepler_clean['mission'] = 'Kepler'
    
    # --- Clean K2 Data ---
    for col in common_columns:
        if col in k2_clean.columns:
            if col == 'disposition':
                k2_clean[col].fillna('UNKNOWN', inplace=True)
            else:
                k2_clean[col].fillna(k2_clean[col].median(), inplace=True)
    
    k2_clean['mission'] = 'K2'
    
    # --- Clean TESS Data ---
    for col in common_columns:
        if col in tess_clean.columns:
            if col == 'disposition':
                tess_clean[col].fillna('UNKNOWN', inplace=True)
            else:
                tess_clean[col].fillna(tess_clean[col].median(), inplace=True)
    
    tess_clean['mission'] = 'TESS'
    
    # Find common columns across all datasets
    kepler_cols = set(kepler_clean.columns)
    k2_cols = set(k2_clean.columns)
    tess_cols = set(tess_clean.columns)
    common_cols = list(kepler_cols & k2_cols & tess_cols)
    
    # If no common columns, use our standard columns
    if not common_cols:
        common_cols = ['pl_orbper', 'pl_rade', 'pl_eqt', 'st_teff', 'st_logg', 'st_rad', 'disposition', 'mission']
    
    # Select only common columns and merge
    kepler_final = kepler_clean[common_cols]
    k2_final = k2_clean[common_cols]
    tess_final = tess_clean[common_cols]
    
    master_df = pd.concat([kepler_final, k2_final, tess_final], ignore_index=True)
    
    return master_df

def create_sample_combined_data():
    """Create combined sample data"""
    np.random.seed(45)
    n_samples = 2000
    
    missions = ['Kepler'] * 800 + ['K2'] * 600 + ['TESS'] * 600
    np.random.shuffle(missions)
    
    data = {
        'pl_orbper': np.random.exponential(30, n_samples),
        'pl_rade': np.random.lognormal(0.3, 0.7, n_samples),
        'pl_eqt': np.random.normal(1500, 500, n_samples),
        'st_teff': np.random.normal(5300, 600, n_samples),
        'st_logg': np.random.normal(4.5, 0.3, n_samples),
        'st_rad': np.random.normal(0.9, 0.3, n_samples),
        'disposition': np.random.choice(['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'], n_samples, p=[0.3, 0.4, 0.3]),
        'mission': missions
    }
    return pd.DataFrame(data)

def smart_data_doctor(new_data):
    """Automatically fixes common data problems"""
    data_fixes = []
    
    try:
        # 1. Fix column names
        column_mapping = {
            'pl_orbper': ['pl_orbper', 'koi_period', 'period'],
            'pl_rade': ['pl_rade', 'koi_prad', 'radius'],
            'st_teff': ['st_teff', 'koi_steff', 'teff'],
            'st_rad': ['st_rad', 'koi_srad', 'srad'],
            'st_logg': ['st_logg', 'koi_slogg', 'logg'],
            'pl_eqt': ['pl_eqt', 'teq'],
            'disposition': ['disposition', 'koi_disposition', 'status']
        }
        
        for standard_name, possible_names in column_mapping.items():
            for name in possible_names:
                if name in new_data.columns and standard_name not in new_data.columns:
                    new_data[standard_name] = new_data[name]
                    data_fixes.append(f"Renamed '{name}' to '{standard_name}'")
                    break
        
        # 2. Handle missing values
        for col in new_data.columns:
            if new_data[col].dtype in ['float64', 'int64']:
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
                    else:
                        new_data[col].fillna(new_data[col].median(), inplace=True)
                    data_fixes.append(f"Filled {missing_count} missing values in '{col}'")
        
        return new_data, data_fixes
        
    except Exception as e:
        st.error(f"❌ Error in data cleaning: {str(e)}")
        return new_data.fillna(0), [f"Error: {str(e)}"]

def detect_target_column(data):
    """Detect if target labels exist in the data"""
    target_indicators = ['disposition', 'koi_disposition', 'label', 'target']
    
    for col in target_indicators:
        if col in data.columns:
            unique_vals = data[col].dropna().unique()
            if len(unique_vals) > 1:
                return col, unique_vals
    
    return None, None

def create_sample_data():
    """Create sample data for testing"""
    np.random.seed(46)
    sample_data = {
        'pl_orbper': np.random.uniform(1, 365, 50),
        'pl_rade': np.random.uniform(0.5, 20, 50),
        'st_teff': np.random.uniform(3000, 7000, 50),
        'st_rad': np.random.uniform(0.5, 2.0, 50),
        'st_logg': np.random.uniform(4.0, 5.0, 50),
        'disposition': np.random.choice(['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'], 50)
    }
    return pd.DataFrame(sample_data)

class RealMLModels:
    """Real ML Models for Exoplanet Classification"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = ['pl_orbper', 'pl_rade', 'st_teff', 'st_rad', 'st_logg']
        self.trained = False
        self.performance_metrics = {}
    
    def prepare_features(self, data):
        """Prepare features for ML models"""
        features_df = data.copy()
        
        # Ensure all required columns exist
        for col in self.feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0
        
        # Handle missing values
        features_df = features_df[self.feature_columns].fillna(0)
        
        return features_df
    
    def prepare_target(self, data):
        """Prepare target variable for classification"""
        if 'disposition' in data.columns:
            # Convert disposition to binary (1 for planets, 0 for non-planets)
            data['target'] = data['disposition'].apply(
                lambda x: 1 if 'CONFIRMED' in str(x).upper() or 'CANDIDATE' in str(x).upper() else 0
            )
            return data['target']
        return None
    
    def train_models(self, data):
        """Train real ML models on the data"""
        try:
            # Prepare features and target
            X = self.prepare_features(data)
            y = self.prepare_target(data)
            
            if y is None or len(y.unique()) < 2:
                st.warning("⚠ Not enough labeled data to train models")
                return False
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.scalers['standard'] = scaler
            
            # Train Random Forest
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train_scaled, y_train)
            self.models['Random Forest'] = rf_model
            
            # Train SVM
            svm_model = SVC(probability=True, random_state=42)
            svm_model.fit(X_train_scaled, y_train)
            self.models['SVM'] = svm_model
            
            # Calculate performance metrics
            self._calculate_performance_metrics(X_test_scaled, y_test)
            
            self.trained = True
            return True
            
        except Exception as e:
            st.error(f"❌ Error training models: {str(e)}")
            return False
    
    def _calculate_performance_metrics(self, X_test, y_test):
        """Calculate performance metrics for trained models"""
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            self.performance_metrics[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
    
    def predict(self, data, model_name):
        """Make predictions using specified model"""
        if not self.trained or model_name not in self.models:
            return None, None
        
        X = self.prepare_features(data)
        X_scaled = self.scalers['standard'].transform(X)
        
        model = self.models[model_name]
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]
        
        return predictions, probabilities

class ModelSimulator:
    """Simulate different ML models for exoplanet prediction"""
    
    def __init__(self, real_models=None):
        self.real_models = real_models
    
    def predict_random_forest(self, data):
        """Use real Random Forest or simulate predictions"""
        if self.real_models and self.real_models.trained:
            return self.real_models.predict(data, 'Random Forest')
        else:
            return self._simulate_random_forest(data)
    
    def predict_svm(self, data):
        """Use real SVM or simulate predictions"""
        if self.real_models and self.real_models.trained:
            return self.real_models.predict(data, 'SVM')
        else:
            return self._simulate_svm(data)
    
    def predict_quantum_hybrid(self, data):
        """Simulate Quantum-Hybrid predictions"""
        return self._simulate_quantum_hybrid(data)
    
    def _simulate_random_forest(self, data):
        """Simulate Random Forest predictions"""
        predictions = []
        confidences = []
        
        for _, row in data.iterrows():
            score = 0
            
            # Random Forest-like rules
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
            
            score += np.random.normal(0, 0.05)
            score = max(0, min(1, score))
            
            prediction = 1 if score > 0.5 else 0
            confidence = score if prediction == 1 else 1 - score
            
            predictions.append(prediction)
            confidences.append(confidence)
        
        return predictions, confidences
    
    def _simulate_svm(self, data):
        """Simulate SVM predictions"""
        predictions = []
        confidences = []
        
        for _, row in data.iterrows():
            score = 0
            
            if 0.8 <= row.get('pl_rade', 0) <= 2.5:
                score += 0.4
            if 100 <= row.get('pl_orbper', 0) <= 300:
                score += 0.3
            if 5500 <= row.get('st_teff', 0) <= 6000:
                score += 0.2
            if 0.9 <= row.get('st_rad', 0) <= 1.1:
                score += 0.1
            
            score = max(0, min(1, score))
            prediction = 1 if score > 0.6 else 0
            confidence = score if prediction == 1 else 1 - score
            
            predictions.append(prediction)
            confidences.append(confidence)
        
        return predictions, confidences
    
    def _simulate_quantum_hybrid(self, data):
        """Simulate Quantum-Hybrid predictions"""
        predictions = []
        confidences = []
        
        for _, row in data.iterrows():
            features = [
                row.get('pl_rade', 0),
                row.get('pl_orbper', 0),
                row.get('st_teff', 0),
                row.get('st_rad', 0),
                row.get('st_logg', 0)
            ]
            
            # Simplified quantum-inspired scoring
            score = sum(f * 0.2 for f in features if f > 0) / 5
            
            if 0.5 <= row.get('pl_rade', 0) <= 4.0:
                score *= 1.2
            if 10 <= row.get('pl_orbper', 0) <= 500:
                score *= 1.1
            
            score = max(0, min(1, score))
            
            prediction = 1 if score > 0.45 else 0
            confidence = score if prediction == 1 else 1 - score
            
            predictions.append(prediction)
            confidences.append(confidence)
        
        return predictions, confidences

def plot_performance_comparison(metrics_dict):
    """Create performance comparison visualization"""
    if not metrics_dict:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    models = list(metrics_dict.keys())
    
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_keys = ['accuracy', 'precision', 'recall', 'f1']
    
    for idx, (metric_name, metric_key) in enumerate(zip(metric_names, metric_keys)):
        ax = axes[idx//2, idx%2]
        
        values = [metrics_dict[model][metric_key] for model in models]
        
        bars = ax.bar(models, values, color=['#4CAF50', '#2196F3', '#9C27B0'], alpha=0.8)
        ax.set_title(f'{metric_name} Comparison')
        ax.set_ylabel(metric_name)
        ax.set_ylim(0, 1)
        
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

# Main app execution
def main():
    # Load the data
    with st.spinner('Loading exoplanet data... 🚀'):
        kepler_df, k2_df, tess_df = load_data()
    
    # Clean and merge data
    with st.spinner('Cleaning and merging data... 🔄'):
        master_df = clean_and_merge_data(kepler_df, k2_df, tess_df)
    
    if not master_df.empty:
        st.success(f"🎉 Successfully loaded {len(master_df)} exoplanets from all missions!")
        
        # Initialize and train real ML models
        st.header("🤖 Real ML Model Training")
        
        real_ml_models = RealMLModels()
        
        if st.button("🚀 Train Real ML Models", type="primary"):
            with st.spinner("Training real ML models on your data..."):
                success = real_ml_models.train_models(master_df)
                
                if success:
                    st.success("✅ Real ML models trained successfully!")
                    # Show performance metrics
                    if real_ml_models.performance_metrics:
                        st.subheader("📊 Model Performance")
                        for model_name, metrics in real_ml_models.performance_metrics.items():
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                            with col2:
                                st.metric("Precision", f"{metrics['precision']:.3f}")
                            with col3:
                                st.metric("Recall", f"{metrics['recall']:.3f}")
                            with col4:
                                st.metric("F1-Score", f"{metrics['f1']:.3f}")
                else:
                    st.warning("Using simulated models as fallback")
        
        # Show mission distribution
        if 'mission' in master_df.columns:
            mission_counts = master_df['mission'].value_counts()
            
            # Main content
            st.header("📊 Dataset Overview")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Objects", len(master_df))
            with col2: 
                st.metric("Number of Features", len(master_df.columns))
            with col3:
                st.metric("Different Missions", len(master_df['mission'].unique()))
            
            # Mission distribution chart
            st.subheader("Objects Found by Each Telescope")
            fig, ax = plt.subplots(figsize=(8, 5))
            mission_counts.plot(kind='bar', color=['skyblue', 'lightcoral', 'lightgreen'], ax=ax)
            plt.title('Number of Objects by Mission')
            plt.xlabel('Mission')
            plt.ylabel('Count')
            plt.xticks(rotation=0)
            
            for i, v in enumerate(mission_counts):
                plt.text(i, v + 50, str(v), ha='center', va='bottom')
            
            st.pyplot(fig)
        
        # Data Doctor Section
        st.header("🏥 Data Doctor")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload Your Own Data")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    user_data = pd.read_csv(uploaded_file)
                    st.success(f"✅ Uploaded {len(user_data)} rows")
                    
                    if st.button("🩺 Run Data Doctor"):
                        with st.spinner("Cleaning your data..."):
                            cleaned_data, fixes = smart_data_doctor(user_data.copy())
                        
                        st.subheader("🧹 Cleaning Results")
                        if fixes:
                            for fix in fixes:
                                st.write(f"• {fix}")
                        
                        st.dataframe(cleaned_data.head(10))
                        
                        csv = cleaned_data.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Cleaned Data",
                            data=csv,
                            file_name="cleaned_exoplanet_data.csv",
                            mime="text/csv",
                        )
                        
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        # Model Testing Section
        st.header("🧪 Model Testing Laboratory")
        
        model_simulator = ModelSimulator(real_ml_models)
        
        use_sample = st.checkbox("🎯 Use sample data for testing", value=True)
        
        if use_sample:
            test_data = create_sample_data()
            st.info("🎯 Using sample data for demonstration")
            
            if st.button("🚀 Run Model Tests"):
                st.subheader("📈 Model Performance Results")
                
                # Test all models
                models = {
                    'Random Forest': model_simulator.predict_random_forest,
                    'SVM': model_simulator.predict_svm,
                    'Quantum-Hybrid': model_simulator.predict_quantum_hybrid
                }
                
                results = {}
                for model_name, predict_func in models.items():
                    predictions, confidences = predict_func(test_data)
                    planet_count = sum(predictions)
                    
                    results[model_name] = {
                        'predictions': predictions,
                        'confidences': confidences,
                        'planet_count': planet_count
                    }
                
                # Display results
                for model_name, result in results.items():
                    model_type = "🧠 REAL MODEL" if (real_ml_models.trained and model_name in ['Random Forest', 'SVM']) else "🎭 SIMULATED"
                    
                    st.markdown(f"""
                    <div class="model-card {'quantum-card' if 'Quantum' in model_name else 'svm-card' if 'SVM' in model_name else ''}">
                        <h3>{model_type} - {model_name}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.info(f"🪐 {model_name} found {result['planet_count']} planets out of {len(test_data)} objects")
        
        # Explore Planets Section
        st.header("🔍 Explore the Planets")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'mission' in master_df.columns:
                selected_mission = st.selectbox(
                    "Choose a telescope mission:",
                    ["All"] + list(master_df['mission'].unique())
                )
            else:
                selected_mission = "All"
                st.info("No mission data available")
            
        with col2:
            max_radius = st.slider(
                "Maximum Planet Size (Earth radii):",
                min_value=0.1,
                max_value=50.0,
                value=20.0,
                step=1.0
            )
        
        # Filter data
        filtered_df = master_df.copy()
        
        if selected_mission != "All" and 'mission' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['mission'] == selected_mission]
        
        if 'pl_rade' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['pl_rade'] <= max_radius]
        
        st.write(f"Showing {len(filtered_df)} objects")
        
        # Charts
        st.header("📈 Planet Characteristics")
        
        tab1, tab2, tab3 = st.tabs(["Planet Sizes", "Orbital Periods", "Temperatures"])
        
        with tab1:
            if 'pl_rade' in filtered_df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                radius_data = filtered_df['pl_rade'].dropna()
                if len(radius_data) > 0:
                    ax.hist(radius_data, bins=30, alpha=0.7, edgecolor='black')
                    ax.set_xlabel('Planet Radius [Earth Radii]')
                    ax.set_ylabel('Count')
                    ax.set_title('Planet Size Distribution')
                    st.pyplot(fig)
                else:
                    st.info("No planet radius data available")
        
        with tab2:
            if 'pl_orbper' in filtered_df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                period_data = filtered_df['pl_orbper'].dropna()
                if len(period_data) > 0:
                    period_data_log = np.log10(period_data[period_data > 0])
                    ax.hist(period_data_log, bins=30, alpha=0.7, edgecolor='black')
                    ax.set_xlabel('log10(Orbital Period [days])')
                    ax.set_ylabel('Count')
                    ax.set_title('Orbital Period Distribution')
                    st.pyplot(fig)
                else:
                    st.info("No orbital period data available")
        
        # Data Table
        st.header("📋 Planet Data Table")
        num_rows = st.slider("Number of objects to show:", 5, 100, 10)
        st.dataframe(filtered_df.head(num_rows))
        
        # Fun Facts
        st.header("🎉 Fun Facts")
        
        if not filtered_df.empty:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'pl_rade' in filtered_df.columns:
                    biggest_planet = filtered_df['pl_rade'].max()
                    st.metric("Largest Planet", f"{biggest_planet:.1f} Earth radii")
                else:
                    st.metric("Largest Planet", "N/A")
            
            with col2:
                if 'pl_orbper' in filtered_df.columns:
                    longest_year = filtered_df['pl_orbper'].max()
                    st.metric("Longest Year", f"{longest_year:.0f} days")
                else:
                    st.metric("Longest Year", "N/A")
            
            with col3:
                if 'pl_eqt' in filtered_df.columns:
                    hottest_planet = filtered_df['pl_eqt'].max()
                    st.metric("Hottest Planet", f"{hottest_planet:.0f} K")
                else:
                    st.metric("Hottest Planet", "N/A")
        
        # Footer
        st.markdown("---")
        st.markdown("### 🚀 About This App")
        st.markdown("Explore exoplanet data with real machine learning models! This app works with real NASA data files or generates realistic sample data for demonstration.")
        
    else:
        st.error("❌ Could not load any data. Please check your data files or try the sample data features.")

# Run the main function
if __name__ == "__main__":
    main()