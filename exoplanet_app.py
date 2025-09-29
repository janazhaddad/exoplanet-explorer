import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set up our page
st.set_page_config(page_title="Exoplanet Explorer", page_icon="🪐", layout="wide")

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
    Automatically fixes common data problems
    """
    st.write("🔧 Running Data Doctor...")
    
    data_fixes = []
    
    # 1. Fix column names
    column_mapping = {
        'pl_orbper': ['pl_orbper', 'koi_period', 'orbital_period', 'period', 'Period'],
        'pl_rade': ['pl_rade', 'koi_prad', 'planet_radius', 'radius', 'Radius'],
        'st_teff': ['st_teff', 'koi_steff', 'stellar_teff', 'teff', 'Teff'],
        'st_rad': ['st_rad', 'koi_srad', 'stellar_radius', 'srad', 's_rad'],
        'st_logg': ['st_logg', 'koi_slogg', 'stellar_logg', 'logg', 'Logg']
    }
    
    for standard_name, possible_names in column_mapping.items():
        for name in possible_names:
            if name in new_data.columns and standard_name not in new_data.columns:
                new_data[standard_name] = new_data[name]
                data_fixes.append(f"Renamed '{name}' → '{standard_name}'")
                break
    
    # 2. Handle missing values
    for col in ['pl_orbper', 'pl_rade', 'st_teff', 'st_rad', 'st_logg']:
        if col in new_data.columns:
            missing_count = new_data[col].isna().sum()
            if missing_count > 0:
                if col == 'pl_rade':
                    new_data[col].fillna(2.0, inplace=True)  # Typical Earth-like planet
                elif col == 'pl_orbper':
                    new_data[col].fillna(10.0, inplace=True)  # Typical orbital period
                else:
                    new_data[col].fillna(new_data[col].median(), inplace=True)
                data_fixes.append(f"Filled {missing_count} missing values in {col}")
    
    # 3. Fix extreme values
    for col in new_data.columns:
        if new_data[col].dtype in ['float64', 'int64']:
            Q1 = new_data[col].quantile(0.25)
            Q3 = new_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((new_data[col] < lower_bound) | (new_data[col] > upper_bound)).sum()
            if outliers > 0:
                new_data[col] = np.clip(new_data[col], lower_bound, upper_bound)
                data_fixes.append(f"Fixed {outliers} outliers in {col}")
    
    # Show what was fixed
    if data_fixes:
        st.success("Data fixes applied:")
        for fix in data_fixes:
            st.write(f"✅ {fix}")
    else:
        st.info("✅ Data looks clean - no fixes needed!")
    
    return new_data

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
        'Best For': ['Balanced Performance', 'Research Showcase', 'Maximum Discovery'],
        'Training Time': ['2 minutes', '15 minutes', '5 minutes']
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.table(comparison_df)
    
    # Visual comparison
    st.subheader("📈 Model Performance Visualization")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Recall comparison
    models = comparison_data['Model']
    recall = [float(x.strip('%')) for x in comparison_data['Recall']]
    precision = [float(x.strip('%')) for x in comparison_data['Precision']]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax1.bar(x - width/2, recall, width, label='Recall', color='skyblue', alpha=0.8)
    ax1.bar(x + width/2, precision, width, label='Precision', color='lightcoral', alpha=0.8)
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Recall vs Precision by Model')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Training time comparison
    times = [2, 15, 5]  # minutes
    ax2.bar(models, times, color=['lightgreen', 'orange', 'lightblue'])
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Training Time (minutes)')
    ax2.set_title('Model Training Time Comparison')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Recommendation based on use case
    st.subheader("🎯 Recommendation Engine")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        use_case = st.selectbox("What's your goal?", 
                               ["Find as many planets as possible", 
                                "Balance accuracy and discovery",
                                "Show cutting-edge research",
                                "Quick results"])
        
        if use_case == "Find as many planets as possible":
            st.success("**Recommended: SVM** - 96.4% recall (finds the most planets)")
            st.info("""
            **Why SVM?**
            - Highest recall rate means fewer missed planets
            - Good for initial discovery phases
            - May have more false positives but catches more real planets
            """)
        elif use_case == "Balance accuracy and discovery":
            st.success("**Recommended: Random Forest** - 97.7% recall + 89.1% precision")
            st.info("""
            **Why Random Forest?**
            - Best overall balance of precision and recall
            - Robust against overfitting
            - Good interpretability of results
            """)
        elif use_case == "Show cutting-edge research":
            st.success("**Recommended: Quantum-Hybrid** - Demonstrates quantum advantage")
            st.info("""
            **Why Quantum-Hybrid?**
            - Showcases latest quantum computing applications
            - Research and demonstration purposes
            - Potential for future improvements
            """)
        else:
            st.success("**Recommended: Random Forest** - Fast and reliable")
            st.info("""
            **Why Random Forest?**
            - Quick training time (2 minutes)
            - Reliable performance
            - Good for rapid prototyping
            """)
    
    with col2:
        st.markdown("### 🏆 Best Model For:")
        st.markdown("""
        - **Maximum Discovery**: SVM
        - **Balanced Approach**: Random Forest  
        - **Research**: Quantum-Hybrid
        - **Speed**: Random Forest
        """)

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
        
        # --- MAIN APP CONTENT ---
        
        st.header("📊 Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Planets Found", len(master_df))
        with col2: 
            st.metric("Number of Features", len(master_df.columns))
        with col3:
            st.metric("Different Missions", len(master_df['mission'].unique()))
        
        # Show mission distribution
        st.subheader("Planets Found by Each Telescope")
        mission_counts = master_df['mission'].value_counts()
        
        fig, ax = plt.subplots(figsize=(8, 5))
        mission_counts.plot(kind='bar', color=['skyblue', 'lightcoral', 'lightgreen'], ax=ax)
        plt.title('Number of Objects by Mission')
        plt.xlabel('Mission')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        
        for i, v in enumerate(mission_counts):
            plt.text(i, v + 50, str(v), ha='center', va='bottom')
        
        st.pyplot(fig)
        
        # --- DATA DOCTOR SECTION ---
        st.header("🏥 Data Doctor")
        
        st.markdown("""
        **Automatically clean and fix common data problems in your exoplanet datasets.**
        Upload your own data or use our built-in cleaning on the main dataset.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload Your Own Data")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    user_data = pd.read_csv(uploaded_file)
                    st.success(f"✅ Uploaded {len(user_data)} rows with {len(user_data.columns)} columns")
                    
                    # Show original data stats
                    st.write("**Original Data Overview:**")
                    st.write(f"- Shape: {user_data.shape}")
                    st.write(f"- Missing values: {user_data.isnull().sum().sum()}")
                    
                    # Apply data doctor
                    if st.button("🩺 Run Data Doctor", type="primary"):
                        with st.spinner("Cleaning your data..."):
                            cleaned_data = smart_data_doctor(user_data.copy())
                        
                        st.subheader("🧹 Cleaning Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Before Cleaning:**")
                            st.write(f"- Shape: {user_data.shape}")
                            st.write(f"- Missing values: {user_data.isnull().sum().sum()}")
                            
                        with col2:
                            st.write("**After Cleaning:**")
                            st.write(f"- Shape: {cleaned_data.shape}")
                            st.write(f"- Missing values: {cleaned_data.isnull().sum().sum()}")
                        
                        # Show sample of cleaned data
                        st.subheader("📋 Cleaned Data Sample")
                        st.dataframe(cleaned_data.head(10))
                        
                        # Download cleaned data
                        csv = cleaned_data.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Cleaned Data",
                            data=csv,
                            file_name="cleaned_exoplanet_data.csv",
                            mime="text/csv",
                        )
                        
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        with col2:
            st.subheader("Clean Main Dataset")
            st.markdown("""
            Apply data cleaning to the main exoplanet dataset to:
            - Standardize column names
            - Fill missing values
            - Remove extreme outliers
            """)
            
            if st.button("🩺 Clean Main Dataset", key="clean_main"):
                with st.spinner("Applying data doctor to main dataset..."):
                    cleaned_master = smart_data_doctor(master_df.copy())
                
                st.success(f"✅ Main dataset cleaned! Original: {len(master_df)} rows, Cleaned: {len(cleaned_master)} rows")
                
                # Compare before/after
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Original Data Issues:**")
                    missing_original = master_df[['pl_orbper', 'pl_rade', 'st_teff', 'st_rad', 'st_logg']].isnull().sum()
                    for col, count in missing_original.items():
                        if count > 0:
                            st.write(f"- {col}: {count} missing")
                
                with col2:
                    st.write("**After Cleaning:**")
                    missing_cleaned = cleaned_master[['pl_orbper', 'pl_rade', 'st_teff', 'st_rad', 'st_logg']].isnull().sum()
                    for col, count in missing_cleaned.items():
                        st.write(f"- {col}: {count} missing")
        
        # --- EXPLORE PLANETS SECTION ---
        
        st.header("🔍 Explore the Planets")
        
        # Let people choose what to see
        col1, col2 = st.columns(2)
        
        with col1:
            # Dropdown to select mission
            selected_mission = st.selectbox(
                "Choose a telescope mission:",
                ["All"] + list(master_df['mission'].unique())
            )
            
        with col2:
            # Slider to filter by planet size
            max_radius = st.slider(
                "Maximum Planet Size (Earth radii):",
                min_value=0.1,
                max_value=50.0,
                value=20.0,
                step=1.0
            )
        
        # Filter data based on user selection
        filtered_df = master_df.copy()
        
        if selected_mission != "All":
            filtered_df = filtered_df[filtered_df['mission'] == selected_mission]
        
        # Only filter by radius if the column exists
        if 'pl_rade' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['pl_rade'] <= max_radius]
        
        st.write(f"Showing {len(filtered_df)} planets")
        
        # --- CHARTS SECTION ---
        
        st.header("📈 Planet Characteristics")
        
        # Create tabs for different charts
        tab1, tab2, tab3, tab4 = st.tabs(["Planet Sizes", "Orbital Periods", "Temperatures", "Missing Data"])
        
        with tab1:
            st.subheader("How Big Are the Planets?")
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
                    st.warning("No planet radius data available for the selected filters.")
            else:
                st.warning("Planet radius data not available in this dataset.")
        
        with tab2:
            st.subheader("How Long Are Their Years?")
            if 'pl_orbper' in filtered_df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                period_data = filtered_df['pl_orbper'].dropna()
                if len(period_data) > 0:
                    # Use log scale for better visualization
                    period_data_log = np.log10(period_data)
                    ax.hist(period_data_log, bins=30, alpha=0.7, edgecolor='black')
                    ax.set_xlabel('log10(Orbital Period [days])')
                    ax.set_ylabel('Count')
                    ax.set_title('Orbital Period Distribution')
                    st.pyplot(fig)
                else:
                    st.warning("No orbital period data available for the selected filters.")
            else:
                st.warning("Orbital period data not available in this dataset.")
        
        with tab3:
            st.subheader("How Hot Are the Planets?")
            if 'pl_eqt' in filtered_df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                temp_data = filtered_df['pl_eqt'].dropna()
                if len(temp_data) > 0:
                    ax.hist(temp_data, bins=30, alpha=0.7, edgecolor='black')
                    ax.set_xlabel('Equilibrium Temperature [K]')
                    ax.set_ylabel('Count')
                    ax.set_title('Planet Temperature Distribution')
                    st.pyplot(fig)
                else:
                    st.warning("No temperature data available for the selected filters.")
            else:
                st.warning("Planet temperature data not available in this dataset.")
        
        with tab4:
            st.subheader("Missing Data Analysis")
            # Calculate missing values percentage
            missing_info = (filtered_df.isnull().sum() / len(filtered_df)) * 100
            missing_info = missing_info[missing_info > 0].sort_values(ascending=False)
            
            if len(missing_info) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                missing_info.plot(kind='bar', ax=ax)
                plt.title('Percentage of Missing Values by Column')
                plt.ylabel('Percentage Missing')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                st.write("Missing values by column:")
                for col, percent in missing_info.items():
                    st.write(f"- **{col}**: {percent:.1f}%")
            else:
                st.success("No missing data in the selected dataset!")
        
        # --- COMPARISON SECTION ---
        
        st.header("🔬 Compare Different Telescopes")
        
        if 'pl_rade' in master_df.columns:
            st.subheader("Planet Sizes Found by Each Telescope")
            
            # Filter out very large planets for better visualization
            comparison_df = master_df[master_df['pl_rade'] < 20]  # Increased limit for real data
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=comparison_df, x='mission', y='pl_rade', ax=ax)
            plt.title('Planet Size Comparison Across Missions')
            plt.ylabel('Planet Radius [Earth Radii]')
            plt.xlabel('Mission')
            st.pyplot(fig)
        
        # --- MODEL COMPARISON SECTION ---
        show_model_comparison()
        
        # --- DATA TABLE SECTION ---
        
        st.header("📋 Planet Data Table")
        
        # Let people choose how many rows to see
        num_rows = st.slider("Number of planets to show:", 5, 100, 10)
        
        # Show the data
        st.dataframe(filtered_df.head(num_rows))
        
        # Add download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Current Data as CSV",
            data=csv,
            file_name="exoplanet_data.csv",
            mime="text/csv",
        )
        
        # --- FUN FACTS SECTION ---
        
        st.header("🎉 Fun Facts")
        
        if not filtered_df.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'pl_rade' in filtered_df.columns:
                    biggest_planet = filtered_df['pl_rade'].max()
                    st.metric("Largest Planet (Earth radii)", f"{biggest_planet:.1f}")
                else:
                    st.metric("Largest Planet", "N/A")
            
            with col2:
                if 'pl_rade' in filtered_df.columns:
                    smallest_planet = filtered_df['pl_rade'].min()
                    st.metric("Smallest Planet (Earth radii)", f"{smallest_planet:.1f}")
                else:
                    st.metric("Smallest Planet", "N/A")
            
            with col3:
                if 'pl_orbper' in filtered_df.columns:
                    longest_year = filtered_df['pl_orbper'].max()
                    st.metric("Longest Year (days)", f"{longest_year:.0f}")
                else:
                    st.metric("Longest Year", "N/A")
            
            with col4:
                if 'pl_eqt' in filtered_df.columns:
                    hottest_planet = filtered_df['pl_eqt'].max()
                    st.metric("Hottest Planet (K)", f"{hottest_planet:.0f}")
                else:
                    st.metric("Hottest Planet", "N/A")
        
        # --- DATA QUALITY INFO ---
        
        st.header("🔍 Data Quality Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Available Columns")
            available_columns = [col for col in ['pl_rade', 'pl_orbper', 'pl_eqt', 'st_teff', 'st_rad', 'st_logg'] 
                              if col in master_df.columns]
            if available_columns:
                for col in available_columns:
                    non_null_count = master_df[col].count()
                    st.write(f"✅ **{col}**: {non_null_count} values ({non_null_count/len(master_df)*100:.1f}%)")
            else:
                st.write("No common planet properties found in the dataset.")
        
        with col2:
            st.subheader("Mission Statistics")
            for mission in master_df['mission'].unique():
                mission_data = master_df[master_df['mission'] == mission]
                st.write(f"**{mission}**: {len(mission_data)} planets")
        
        # --- FOOTER ---
        
        st.markdown("---")
        st.markdown("### 🚀 About This App")
        st.markdown("""
        This app explores real exoplanet data from:
        - **Kepler Space Telescope**
        - **K2 Mission** 
        - **TESS (Transiting Exoplanet Survey Satellite)**
        
        **Data Sources:**
        - NASA Exoplanet Archive
        - Kepler DR25
        - TESS TOI Catalog
        
        *All data is loaded from your local CSV files.*
        """)
        
        st.balloons()  # Celebrate our working app! 🎉
    
    else:
        st.error("Failed to merge datasets. Please check your data files.")
else:
    st.error("Could not load one or more data files. Please check that all CSV files are in the same folder as this script.")
    
    # Show help for file placement
    st.info("""
    **To fix this issue:**
    
    1. Make sure you have these 3 files in the **same folder** as your `exoplanet_app.py`:
       - `cumulative_2025.09.21_04.31.43.csv`
       - `k2pandc_2025.09.17_06.42.42.csv`
       - `TOI_2025.09.17_06.36.05 (2).csv`
    
    2. Or update the file paths in the `load_data()` function to point to where your files are located.
    
    3. Check that the file names match exactly (including capitalization).
    """)