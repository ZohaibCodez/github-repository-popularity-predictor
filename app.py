import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="GitHub Repository Popularity Predictor",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI/UX
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Headers */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        animation: fadeInDown 1s ease-in;
    }
    
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        color: white;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.2);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Input Fields */
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        transition: border-color 0.3s ease;
    }
    
    .stNumberInput>div>div>input:focus,
    .stSelectbox>div>div>select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    /* Cards */
    .prediction-card {
        background: white;
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid #f0f0f0;
        transition: all 0.3s ease;
    }
    
    .prediction-card:hover {
        box-shadow: 0 15px 50px rgba(0,0,0,0.15);
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Alert Boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid;
        animation: fadeIn 0.5s ease;
    }
    
    /* Info boxes */
    div[data-testid="stMarkdownContainer"] p {
        line-height: 1.8;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Success/Warning boxes */
    .element-container {
        animation: fadeIn 0.5s ease;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    /* Prediction result box */
    .prediction-result {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 3rem;
        border-radius: 25px;
        text-align: center;
        color: white;
        font-size: 3rem;
        font-weight: 700;
        box-shadow: 0 20px 60px rgba(245, 87, 108, 0.4);
        animation: fadeIn 0.8s ease;
    }
    
    /* Category badges */
    .category-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 0.5rem;
    }
    
    .badge-growing {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
    }
    
    .badge-popular {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    
    .badge-viral {
        background: linear-gradient(135deg, #30cfd0 0%, #330867 100%);
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    """Load the dataset"""
    try:
        df = pd.read_csv('data/repositories.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load models
@st.cache_resource
def load_models():
    """Load trained models and preprocessors"""
    models = {}
    try:
        models_dir = Path('models')
        if models_dir.exists():
            if (models_dir / 'best_model.pkl').exists():
                models['best_model'] = joblib.load('models/best_model.pkl')
                models['scaler'] = joblib.load('models/scaler.pkl')
                models['label_encoders'] = joblib.load('models/label_encoders.pkl')
                models['feature_columns'] = joblib.load('models/feature_columns.pkl')
                models['metadata'] = joblib.load('models/model_metadata.pkl')
    except Exception as e:
        st.warning(f"Models not loaded: {e}")
    return models

def main():
    # Enhanced Sidebar
    st.sidebar.markdown('''
    <div style="text-align: center; padding: 1.5rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); margin: -1rem -1rem 2rem -1rem; border-radius: 0 0 20px 20px;">
        <h1 style="color: white; font-size: 1.8rem; margin: 0;">⭐ GitHub Predictor</h1>
        <p style="color: white; opacity: 0.9; font-size: 0.9rem; margin: 0.5rem 0 0 0;">ML-Powered Analysis</p>
    </div>
    ''', unsafe_allow_html=True)
    
    st.sidebar.markdown("### 🧭 Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["🏠 Home", "📈 EDA", "🤖 Model & Predictions", "📋 Conclusion"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Quick Stats in Sidebar
    st.sidebar.markdown("### 📊 Quick Stats")
    st.sidebar.info("""
    **Dataset:** 215K+ repos  
    **Models:** 6 algorithms  
    **Features:** 24 attributes  
    **Analyses:** 15+ EDA
    """)
    
    st.sidebar.markdown("---")
    
    # Project Info
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("""
    **Course:** IDS F24  
    **Author:** Zohaib Khan  
    **Year:** 2025
    """)
    
    # Load data
    df = load_data()
    models = load_models()
    
    if page == "🏠 Home":
        show_home(df)
    elif page == "📈 EDA":
        show_eda(df)
    elif page == "🤖 Model & Predictions":
        show_model(df, models)
    elif page == "📋 Conclusion":
        show_conclusion(df, models)

def show_home(df):
    """Home page"""
    # Animated header
    st.markdown('''
    <div style="text-align: center; animation: fadeInDown 1s ease;">
        <h1 style="font-size: 4rem; font-weight: 700; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;">
            ⭐ GitHub Repository Popularity Predictor
        </h1>
        <p style="font-size: 1.3rem; color: #666; margin-bottom: 3rem;">
            Predicting GitHub Repository Stars using Machine Learning ✨
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Quick stats banner
    st.markdown('<div style="animation: fadeIn 1s ease;">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('''
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; text-align: center; color: white; box-shadow: 0 10px 30px rgba(102,126,234,0.3);">
            <h2 style="margin: 0; font-size: 2.5rem;">215K+</h2>
            <p style="margin: 0; font-size: 1rem; opacity: 0.9;">Repositories</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown('''
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 15px; text-align: center; color: white; box-shadow: 0 10px 30px rgba(245,87,108,0.3);">
            <h2 style="margin: 0; font-size: 2.5rem;">15+</h2>
            <p style="margin: 0; font-size: 1rem; opacity: 0.9;">EDA Analyses</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown('''
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1.5rem; border-radius: 15px; text-align: center; color: white; box-shadow: 0 10px 30px rgba(79,172,254,0.3);">
            <h2 style="margin: 0; font-size: 2.5rem;">6</h2>
            <p style="margin: 0; font-size: 1rem; opacity: 0.9;">ML Models</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        st.markdown('''
        <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 1.5rem; border-radius: 15px; text-align: center; color: white; box-shadow: 0 10px 30px rgba(250,112,154,0.3);">
            <h2 style="margin: 0; font-size: 2.5rem;">24</h2>
            <p style="margin: 0; font-size: 1rem; opacity: 0.9;">Features</p>
        </div>
        ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Project Overview
    st.markdown('''
    <div style="background: white; padding: 2rem; border-radius: 20px; box-shadow: 0 10px 40px rgba(0,0,0,0.1); margin: 2rem 0; animation: fadeIn 1.2s ease;">
        <h2 style="color: #667eea; margin-bottom: 1rem;">🎯 Project Overview</h2>
        <p style="font-size: 1.1rem; line-height: 1.8; color: #555;">
            This comprehensive machine learning application analyzes and predicts GitHub repository popularity. 
            Using a dataset of <strong>215,029+ repositories</strong> from Kaggle, we've developed predictive models 
            that can estimate how many stars a repository will receive based on various characteristics.
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Key Features Cards
    st.markdown('<h2 style="text-align: center; color: #667eea; margin: 3rem 0 2rem 0;">✨ What Makes This Special?</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('''
        <div style="background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%); padding: 2rem; border-radius: 20px; border: 2px solid #667eea40; transition: all 0.3s ease; animation: slideInRight 1s ease;">
            <div style="font-size: 3rem; text-align: center; margin-bottom: 1rem;">📊</div>
            <h3 style="color: #667eea; text-align: center; margin-bottom: 1rem;">Comprehensive EDA</h3>
            <p style="color: #555; text-align: center; line-height: 1.6;">
                15+ in-depth analyses including distributions, correlations, outlier detection, and feature relationships.
            </p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown('''
        <div style="background: linear-gradient(135deg, #f093fb20 0%, #f5576c20 100%); padding: 2rem; border-radius: 20px; border: 2px solid #f5576c40; transition: all 0.3s ease; animation: slideInRight 1.2s ease;">
            <div style="font-size: 3rem; text-align: center; margin-bottom: 1rem;">🤖</div>
            <h3 style="color: #f5576c; text-align: center; margin-bottom: 1rem;">Smart Predictions</h3>
            <p style="color: #555; text-align: center; line-height: 1.6;">
                Real-time predictions with interactive input forms. Get instant star count estimates with confidence levels.
            </p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown('''
        <div style="background: linear-gradient(135deg, #4facfe20 0%, #00f2fe20 100%); padding: 2rem; border-radius: 20px; border: 2px solid #4facfe40; transition: all 0.3s ease; animation: slideInRight 1.4s ease;">
            <div style="font-size: 3rem; text-align: center; margin-bottom: 1rem;">📈</div>
            <h3 style="color: #4facfe; text-align: center; margin-bottom: 1rem;">Model Comparison</h3>
            <p style="color: #555; text-align: center; line-height: 1.6;">
                Compare 6 different ML algorithms with detailed performance metrics and visualizations.
            </p>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # How it works section
    st.markdown('''
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 3rem; border-radius: 25px; color: white; margin: 2rem 0; animation: fadeIn 1.5s ease;">
        <h2 style="text-align: center; margin-bottom: 2rem; color: white;">🚀 How It Works</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 2rem;">
            <div style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">1️⃣</div>
                <h4 style="color: white;">Data Collection</h4>
                <p style="opacity: 0.9;">215K+ GitHub repositories analyzed</p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">2️⃣</div>
                <h4 style="color: white;">EDA & Preprocessing</h4>
                <p style="opacity: 0.9;">Clean, transform, and prepare data</p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">3️⃣</div>
                <h4 style="color: white;">Model Training</h4>
                <p style="opacity: 0.9;">Train 6 ML algorithms</p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">4️⃣</div>
                <h4 style="color: white;">Predictions</h4>
                <p style="opacity: 0.9;">Get real-time star predictions</p>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Call to action
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('''
        <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 15px; animation: fadeIn 2s ease;">
            <h3 style="color: #667eea; margin-bottom: 1rem;">Ready to Explore? 🎯</h3>
            <p style="color: #555; font-size: 1.1rem; margin-bottom: 1.5rem;">
                Navigate through the sidebar to explore data insights, view EDA results, or make your own predictions!
            </p>
            <p style="color: #764ba2; font-weight: 600; font-size: 1.2rem;">👈 Use the sidebar to get started</p>
        </div>
        ''', unsafe_allow_html=True)
    
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Repositories", f"{len(df):,}")
        with col2:
            st.metric("Languages", f"{df['Language'].nunique():,}")
        with col3:
            st.metric("Avg Stars", f"{df['Stars'].mean():.0f}")
        with col4:
            st.metric("Max Stars", f"{df['Stars'].max():,}")
    
    # Project Goals
    st.markdown('<p class="sub-header">🎯 Project Goals</p>', unsafe_allow_html=True)
    
    goals = """
    1. **Exploratory Data Analysis**: Understand data distributions, correlations, and patterns
    2. **Data Preprocessing**: Handle missing values, outliers, and feature engineering
    3. **Model Development**: Train and compare multiple ML models
    4. **Prediction System**: Enable real-time predictions for new repositories
    5. **Interactive Visualization**: Present insights through an intuitive web interface
    """
    st.markdown(goals)
    
    # Navigation Guide
    st.info("👈 **Use the sidebar** to navigate through different sections of this application!")

def show_eda(df):
    """EDA page with comprehensive analysis"""
    st.markdown('<p class="main-header">📈 Exploratory Data Analysis</p>', unsafe_allow_html=True)
    
    if df is None:
        st.error("Data not loaded!")
        return
    
    # Sidebar filters
    st.sidebar.markdown("### 🔍 Filters")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis",
        ["Dataset Overview", "Univariate Analysis", "Bivariate Analysis", 
         "Correlation Analysis", "Distribution Analysis", "Language Analysis",
         "Missing Values", "Outlier Analysis"]
    )
    
    if analysis_type == "Dataset Overview":
        show_dataset_overview(df)
    elif analysis_type == "Univariate Analysis":
        show_univariate_analysis(df)
    elif analysis_type == "Bivariate Analysis":
        show_bivariate_analysis(df)
    elif analysis_type == "Correlation Analysis":
        show_correlation_analysis(df)
    elif analysis_type == "Distribution Analysis":
        show_distribution_analysis(df)
    elif analysis_type == "Language Analysis":
        show_language_analysis(df)
    elif analysis_type == "Missing Values":
        show_missing_values(df)
    elif analysis_type == "Outlier Analysis":
        show_outlier_analysis(df)

def show_dataset_overview(df):
    """Dataset overview section"""
    st.markdown("## 📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    with col4:
        st.metric("Duplicates", df.duplicated().sum())
    
    # Data preview
    st.markdown("### 🔍 Data Preview")
    st.dataframe(df.head(10), width="stretch")
    
    # Data types
    st.markdown("### 📋 Data Types")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Column Information**")
        info_df = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes,
            'Non-Null': df.count(),
            'Null': df.isnull().sum()
        })
        st.dataframe(info_df, width="stretch")
    
    with col2:
        st.write("**Summary Statistics**")
        st.dataframe(df.describe().T, width="stretch")

def show_univariate_analysis(df):
    """Univariate analysis section"""
    st.markdown("## 📊 Univariate Analysis")
    
    # Numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    selected_feature = st.selectbox("Select Feature", numerical_cols)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### Distribution of {selected_feature}")
        fig = px.histogram(df, x=selected_feature, nbins=50, 
                          title=f"{selected_feature} Distribution",
                          color_discrete_sequence=['#1f77b4'])
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        st.markdown(f"### Box Plot of {selected_feature}")
        fig = px.box(df, y=selected_feature, 
                    title=f"{selected_feature} Box Plot",
                    color_discrete_sequence=['#ff7f0e'])
        st.plotly_chart(fig, width="stretch")
    
    # Statistics
    st.markdown(f"### 📈 Statistics for {selected_feature}")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Mean", f"{df[selected_feature].mean():.2f}")
    with col2:
        st.metric("Median", f"{df[selected_feature].median():.2f}")
    with col3:
        st.metric("Std Dev", f"{df[selected_feature].std():.2f}")
    with col4:
        st.metric("Min", f"{df[selected_feature].min():.2f}")
    with col5:
        st.metric("Max", f"{df[selected_feature].max():.2f}")

def show_bivariate_analysis(df):
    """Bivariate analysis section"""
    st.markdown("## 📊 Bivariate Analysis")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox("Select X Variable", numerical_cols, index=0)
    with col2:
        y_var = st.selectbox("Select Y Variable", numerical_cols, 
                            index=1 if len(numerical_cols) > 1 else 0)
    
    # Scatter plot
    fig = px.scatter(df.sample(min(10000, len(df))), x=x_var, y=y_var,
                    title=f"{x_var} vs {y_var}",
                    trendline="ols",
                    color_discrete_sequence=['#2ca02c'])
    st.plotly_chart(fig, width="stretch")
    
    # Correlation
    corr = df[[x_var, y_var]].corr().iloc[0, 1]
    st.metric("Correlation Coefficient", f"{corr:.4f}")

def show_correlation_analysis(df):
    """Correlation analysis section"""
    st.markdown("## 📊 Correlation Analysis")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numerical_cols].corr()
    
    # Heatmap
    fig = px.imshow(corr_matrix, 
                   text_auto='.2f',
                   aspect="auto",
                   color_continuous_scale='RdBu_r',
                   title="Correlation Heatmap")
    fig.update_layout(height=600)
    st.plotly_chart(fig, width="stretch")
    
    # Top correlations with Stars
    if 'Stars' in numerical_cols:
        st.markdown("### 🌟 Top Correlations with Stars")
        star_corr = corr_matrix['Stars'].sort_values(ascending=False).drop('Stars')
        
        fig = px.bar(x=star_corr.values, y=star_corr.index,
                    orientation='h',
                    title="Feature Correlations with Stars",
                    color=star_corr.values,
                    color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, width="stretch")

def show_distribution_analysis(df):
    """Distribution analysis section"""
    st.markdown("## 📊 Distribution Analysis")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Multiple distributions
    n_cols = st.slider("Number of features to display", 2, min(6, len(numerical_cols)), 4)
    selected_features = st.multiselect("Select Features", numerical_cols, default=numerical_cols[:n_cols])
    
    if selected_features:
        fig = make_subplots(
            rows=(len(selected_features) + 1) // 2, 
            cols=2,
            subplot_titles=selected_features
        )
        
        for idx, feature in enumerate(selected_features):
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            fig.add_trace(
                go.Histogram(x=df[feature], name=feature, nbinsx=50),
                row=row, col=col
            )
        
        fig.update_layout(height=300 * ((len(selected_features) + 1) // 2), 
                         showlegend=False)
        st.plotly_chart(fig, width="stretch")

def show_language_analysis(df):
    """Language analysis section"""
    st.markdown("## 📊 Language Analysis")
    
    if 'Language' not in df.columns:
        st.warning("Language column not found in dataset!")
        return
    
    # Top languages
    top_n = st.slider("Number of top languages", 5, 20, 10)
    top_langs = df['Language'].value_counts().head(top_n)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Most Popular Languages")
        fig = px.bar(x=top_langs.values, y=top_langs.index,
                    orientation='h',
                    title="Top Languages by Repository Count",
                    color=top_langs.values,
                    color_continuous_scale='Viridis')
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        st.markdown("### Language Distribution (Pie Chart)")
        fig = px.pie(values=top_langs.values, names=top_langs.index,
                    title="Language Distribution")
        st.plotly_chart(fig, width="stretch")
    
    # Stars by language
    if 'Stars' in df.columns:
        st.markdown("### ⭐ Average Stars by Language")
        lang_stats = df.groupby('Language')['Stars'].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False).head(top_n)
        
        fig = px.bar(lang_stats, x=lang_stats.index, y='mean',
                    title="Average Stars by Language",
                    color='mean',
                    color_continuous_scale='Blues')
        st.plotly_chart(fig, width="stretch")

def show_missing_values(df):
    """Missing values analysis"""
    st.markdown("## 📊 Missing Values Analysis")
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing Count': missing.values,
        'Missing %': missing_pct.values
    }).sort_values('Missing Count', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Missing Values Table")
        st.dataframe(missing_df, width="stretch")
    
    with col2:
        st.markdown("### Missing Values Visualization")
        fig = px.bar(missing_df[missing_df['Missing Count'] > 0], 
                    x='Column', y='Missing %',
                    title="Missing Values by Column",
                    color='Missing %',
                    color_continuous_scale='Reds')
        st.plotly_chart(fig, width="stretch")

def show_outlier_analysis(df):
    """Outlier detection analysis"""
    st.markdown("## 📊 Outlier Analysis")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_feature = st.selectbox("Select Feature for Outlier Analysis", numerical_cols)
    
    # IQR method
    Q1 = df[selected_feature].quantile(0.25)
    Q3 = df[selected_feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[selected_feature] < lower_bound) | (df[selected_feature] > upper_bound)]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Outliers", len(outliers))
    with col2:
        st.metric("Outlier %", f"{(len(outliers)/len(df)*100):.2f}%")
    with col3:
        st.metric("IQR", f"{IQR:.2f}")
    
    # Visualization
    fig = px.box(df, y=selected_feature, title=f"Outliers in {selected_feature}")
    st.plotly_chart(fig, width="stretch")

def show_model(df, models):
    """Model and predictions page"""
    st.markdown('<p class="main-header">🤖 Model & Predictions</p>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📊 Model Performance", "🔮 Make Predictions"])
    
    with tab1:
        show_model_performance(models)
    
    with tab2:
        show_predictions(models)

def show_model_performance(models):
    """Display model performance metrics"""
    st.markdown("## 📊 Model Performance")
    
    if not models or 'metadata' not in models:
        st.warning("⚠️ Models not available. Please train the models first by running the preprocessing notebook.")
        return
    
    metadata = models['metadata']
    
    # Display best model info
    st.success(f"🏆 **Best Model:** {metadata['model_name']}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R² Score", f"{metadata['test_r2']:.4f}")
    with col2:
        st.metric("RMSE", f"{metadata['test_rmse']:.4f}")
    with col3:
        st.metric("MAE", f"{metadata['test_mae']:.4f}")
    with col4:
        st.metric("Training Samples", f"{metadata['training_samples']:,}")
    
    # Model comparison
    st.markdown("### 📈 Model Comparison")
    
    # Load all model results if available
    model_results = {
        'Model': ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 
                 'Random Forest', 'Gradient Boosting', 'XGBoost'],
        'Description': [
            'Basic linear model',
            'Linear with L2 regularization',
            'Linear with L1 regularization',
            'Ensemble of decision trees',
            'Boosted decision trees',
            'Optimized gradient boosting'
        ]
    }
    
    # Convert to DataFrame and ensure proper types
    models_df = pd.DataFrame(model_results)
    models_df = models_df.astype(str)  # Convert all to string to avoid Arrow serialization issues
    st.dataframe(models_df, width="stretch")
    
    # Feature importance
    st.markdown("### 🎯 Features Used")
    if 'features' in metadata:
        features_df = pd.DataFrame({
            'Feature': metadata['features']
        })
        st.dataframe(features_df, width="stretch")

def show_predictions(models):
    """Make predictions page"""
    # Enhanced header
    st.markdown('''
    <div style="text-align: center; margin-bottom: 2rem; animation: fadeInDown 0.8s ease;">
        <h1 style="font-size: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🔮 Make Predictions
        </h1>
        <p style="font-size: 1.2rem; color: #666;">Enter repository details and get instant star count predictions</p>
    </div>
    ''', unsafe_allow_html=True)
    
    if not models or 'best_model' not in models:
        st.warning("⚠️ Models not available. Please train the models first.")
        st.info("💡 **To generate models:** Run all cells in the `02_preprocessing_and_modeling.ipynb` notebook.")
        return
    
    # Info banner
    st.markdown('''
    <div style="background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); padding: 1.5rem; border-radius: 15px; border-left: 5px solid #667eea; margin-bottom: 2rem;">
        <strong style="color: #667eea; font-size: 1.1rem;">🎯 How it works:</strong>
        <p style="color: #555; margin-top: 0.5rem; line-height: 1.6;">
            Our AI model analyzes repository metrics (forks, issues, size) and characteristics (language, features) to predict popularity. Try different scenarios below!
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Enhanced example scenarios with icons and descriptions
    st.markdown('''
    <h3 style="text-align: center; color: #667eea; margin-bottom: 1.5rem;">🎲 Quick Test Scenarios</h3>
    <p style="text-align: center; color: #666; margin-bottom: 1.5rem;">Click a scenario to auto-fill values</p>
    ''', unsafe_allow_html=True)
    
    col_ex1, col_ex2, col_ex3, col_ex4 = st.columns(4)
    
    with col_ex1:
        if st.button("🌱 Small Project", width="stretch"):
            st.session_state.example = "small"
    with col_ex2:
        if st.button("⭐ Growing Project", width="stretch"):
            st.session_state.example = "growing"
    with col_ex3:
        if st.button("🔥 Popular Project", width="stretch"):
            st.session_state.example = "popular"
    with col_ex4:
        if st.button("🚀 Viral Project", width="stretch"):
            st.session_state.example = "viral"
    
    # Set default values based on example
    if 'example' not in st.session_state:
        st.session_state.example = "growing"
    
    if st.session_state.example == "small":
        default_forks, default_issues, default_size = 10, 5, 500
    elif st.session_state.example == "growing":
        default_forks, default_issues, default_size = 150, 25, 5000
    elif st.session_state.example == "popular":
        default_forks, default_issues, default_size = 500, 100, 15000
    else:  # viral
        default_forks, default_issues, default_size = 2000, 300, 50000
    
    st.markdown("---")
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Repository Metrics")
        forks = st.number_input("Number of Forks", min_value=0, value=default_forks, step=10, 
                               help="How many times this repo has been forked")
        open_issues = st.number_input("Number of Open Issues", min_value=0, value=default_issues, step=1,
                                     help="Active issues in the repository")
        size = st.number_input("Repository Size (KB)", min_value=1, value=default_size, step=100,
                              help="Total size of repository in kilobytes")
    
    with col2:
        st.markdown("#### ⚙️ Repository Features")
        language = st.selectbox("Programming Language", 
                               ['Python', 'JavaScript', 'Java', 'C++', 'Go', 'Ruby', 'TypeScript', 'C#', 'PHP', 'Swift'],
                               help="Primary programming language")
        st.markdown("**Additional Features:**")
        has_wiki = st.checkbox("Has Wiki", value=True, help="Repository has wiki documentation")
        has_issues = st.checkbox("Has Issues", value=True, help="Issue tracking is enabled")
        has_projects = st.checkbox("Has Projects", value=False, help="Project boards enabled")
    
    if st.button("🔮 Predict Stars", type="primary"):
        try:
            # Prepare input data
            input_data = {
                'Forks': forks,
                'Open Issues': open_issues,
                'Size': size,
                'Language': language,
                'Has Wiki': has_wiki,
                'Has Issues': has_issues,
                'Has Projects': has_projects
            }
            
            # Make prediction
            prediction = make_prediction(input_data, models)
            
            if prediction is not None:
                st.success("✅ **Prediction Complete!**")
                
                # Display prediction with enhanced styling
                st.markdown("---")
                pred_stars = int(prediction)
                st.markdown(f'''
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 3rem; border-radius: 25px; text-align: center; margin: 2rem 0; box-shadow: 0 20px 60px rgba(102,126,234,0.4); animation: fadeIn 0.8s ease;">
                    <h1 style="color: white; font-size: 4rem; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">⭐ {pred_stars:,}</h1>
                    <p style="color: white; font-size: 1.5rem; margin-top: 1rem; opacity: 0.9;">Predicted Stars</p>
                </div>
                ''', unsafe_allow_html=True)
                
                # Category and confidence badges
                col1, col2, col3 = st.columns(3)
                with col1:
                    if prediction < 100:
                        category = "🌱 Growing"
                        badge_color = "linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%)"
                    elif prediction < 1000:
                        category = "⭐ Growing"
                        badge_color = "linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%)"
                    elif prediction < 10000:
                        category = "🔥 Popular"
                        badge_color = "linear-gradient(135deg, #fa709a 0%, #fee140 100%)"
                    else:
                        category = "🚀 Viral"
                        badge_color = "linear-gradient(135deg, #30cfd0 0%, #330867 100%)"
                    
                    st.markdown(f'''
                    <div style="background: {badge_color}; padding: 2rem; border-radius: 15px; text-align: center; color: white; box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
                        <h3 style="margin: 0; color: white;">Category</h3>
                        <p style="font-size: 1.5rem; font-weight: 700; margin: 0.5rem 0 0 0; color: white;">{category}</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col2:
                    # Calculate confidence based on input reasonableness
                    confidence = "High"
                    conf_color = "#4ade80"
                    if forks > 0 or open_issues > 0:
                        confidence = "High"
                        conf_color = "#4ade80"
                    elif forks == 0 and open_issues == 0:
                        confidence = "Medium"
                        conf_color = "#facc15"
                    
                    st.markdown(f'''
                    <div style="background: white; padding: 2rem; border-radius: 15px; text-align: center; border: 3px solid {conf_color}; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
                        <h3 style="margin: 0; color: #333;">Confidence</h3>
                        <p style="font-size: 1.5rem; font-weight: 700; margin: 0.5rem 0 0 0; color: {conf_color};">{confidence}</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col3:
                    percentile_temp = 50  # Will be calculated below
                    st.markdown(f'''
                    <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 2rem; border-radius: 15px; text-align: center; color: white; box-shadow: 0 10px 30px rgba(79,172,254,0.3);">
                        <h3 style="margin: 0; color: white;">Ranking</h3>
                        <p style="font-size: 1.5rem; font-weight: 700; margin: 0.5rem 0 0 0; color: white;">Loading...</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Visualization
                st.markdown("---")
                st.markdown("### 📊 Prediction Context")
                
                # Load dataset for comparison
                try:
                    df = pd.read_csv('data/repositories.csv')
                    median_stars = df['Stars'].median()
                    mean_stars = df['Stars'].mean()
                    percentile = (df['Stars'] < prediction).mean() * 100
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"📈 **Your Repository Prediction:** {int(prediction):,} stars\n\n"
                               f"This places your repository in the **{percentile:.1f}th percentile** of all repositories in our dataset.")
                    with col2:
                        comparison_data = pd.DataFrame({
                            'Metric': ['Your Prediction', 'Dataset Median', 'Dataset Mean'],
                            'Stars': [int(prediction), int(median_stars), int(mean_stars)]
                        })
                        fig = px.bar(comparison_data, x='Metric', y='Stars', 
                                   title='Comparison with Dataset',
                                   color='Stars', color_continuous_scale='Viridis')
                        fig.update_layout(showlegend=False, height=300)
                        st.plotly_chart(fig, width="stretch")
                    
                    # Insights
                    st.markdown("### 💡 Insights & Recommendations")
                    if prediction < median_stars:
                        st.warning(f"⚠️ **Below Average Performance**\n\n"
                                 f"Your predicted stars ({int(prediction):,}) is below the dataset median ({int(median_stars):,}).\n\n"
                                 "**💡 Tips to increase popularity:**\n"
                                 "- 📚 Add comprehensive documentation and README\n"
                                 "- 🤝 Increase community engagement (encourage forks and contributions)\n"
                                 "- 🔄 Maintain regular updates and fix issues promptly\n"
                                 "- 🏷️ Add relevant tags and topics\n"
                                 "- 🎯 Solve a real problem with clear use cases")
                    elif prediction < mean_stars:
                        st.info(f"✅ **Good Performance**\n\n"
                              f"Your prediction ({int(prediction):,} stars) is above the median ({int(median_stars):,}) but below average ({int(mean_stars):.0f}).\n\n"
                              "**Keep improving:** Focus on documentation, examples, and community building!")
                    else:
                        st.success(f"🎉 **Excellent Performance!**\n\n"
                                 f"Your predicted stars ({int(prediction):,}) exceed the dataset average ({int(mean_stars):.0f}).\n\n"
                                 "Your repository shows strong potential for high popularity! 🚀")
                    
                    # Feature Impact Analysis
                    st.markdown("---")
                    st.markdown("### 🔍 What's Driving This Prediction?")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📊 Your Input Values:**")
                        input_summary = pd.DataFrame({
                            'Feature': ['Forks', 'Open Issues', 'Size (KB)', 'Language', 'Has Wiki', 'Has Issues', 'Has Projects'],
                            'Value': [forks, open_issues, size, language, 
                                    '✓' if has_wiki else '✗',
                                    '✓' if has_issues else '✗',
                                    '✓' if has_projects else '✗']
                        })
                        st.dataframe(input_summary, width="stretch", hide_index=True)
                    
                    with col2:
                        st.markdown("**💭 Impact Analysis:**")
                        impacts = []
                        if forks > 100:
                            impacts.append("🟢 High fork count boosts popularity")
                        elif forks > 10:
                            impacts.append("🟡 Moderate fork engagement")
                        else:
                            impacts.append("🔴 Low fork count limits reach")
                        
                        if open_issues > 50:
                            impacts.append("🟢 Active issue discussions")
                        elif open_issues > 5:
                            impacts.append("🟡 Some community activity")
                        else:
                            impacts.append("🔴 Limited community engagement")
                        
                        if size > 10000:
                            impacts.append("🟢 Substantial project size")
                        elif size > 1000:
                            impacts.append("🟡 Average project size")
                        else:
                            impacts.append("🔴 Small project size")
                        
                        if has_wiki and has_issues:
                            impacts.append("🟢 Good documentation setup")
                        
                        for impact in impacts:
                            st.markdown(f"- {impact}")
                    
                    # Final Summary Box
                    st.markdown("---")
                    st.markdown("### 📝 Prediction Summary")
                    
                    summary_color = "blue"
                    if prediction < median_stars:
                        summary_emoji = "🌱"
                        summary_text = f"This repository configuration predicts **{int(prediction):,} stars**, which is below the typical repository. Focus on increasing engagement through forks and active issue management to boost popularity."
                    elif prediction < mean_stars:
                        summary_emoji = "⭐"
                        summary_text = f"This repository configuration predicts **{int(prediction):,} stars**, showing good potential. It's above average and on track for steady growth!"
                    else:
                        summary_emoji = "🎉"
                        summary_text = f"This repository configuration predicts **{int(prediction):,} stars**, which is excellent! This suggests high popularity potential with strong community engagement."
                    
                    st.info(f"{summary_emoji} {summary_text}")
                    
                except Exception as e:
                    st.info(f"Based on the input features, this repository is predicted to receive approximately **{int(prediction):,} stars**.")
                    st.warning(f"Note: Unable to load dataset for comparison. Error: {str(e)}")
                
        except Exception as e:
            st.error(f"❌ Error making prediction: {e}")
            st.info("Please ensure all models are properly trained and saved.")

def make_prediction(input_data, models):
    """Make prediction using the trained model"""
    try:
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Apply log transformation to numerical features
        for col in ['Forks', 'Open Issues', 'Size']:
            if col in input_df.columns:
                input_df[f'{col}_log'] = np.log1p(input_df[col])
        
        # Encode categorical variables
        label_encoders = models['label_encoders']
        for col, encoder in label_encoders.items():
            if col in input_df.columns:
                try:
                    input_df[f'{col}_encoded'] = encoder.transform(input_df[col].astype(str))
                except:
                    # Handle unseen categories
                    input_df[f'{col}_encoded'] = 0
        
        # Select features
        feature_columns = models['feature_columns']
        X = input_df[feature_columns]
        
        # Scale features
        scaler = models['scaler']
        X_scaled = scaler.transform(X)
        
        # Make prediction
        model = models['best_model']
        y_pred_log = model.predict(X_scaled)
        
        # Convert back from log scale
        y_pred = np.expm1(y_pred_log[0])
        
        return max(0, y_pred)
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def show_conclusion(df, models):
    """Conclusion page"""
    st.markdown('<p class="main-header">📋 Conclusion</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Project Summary
    
    This project successfully developed a machine learning system to predict GitHub repository popularity (Stars) 
    based on various repository features.
    
    ### ✅ Key Achievements
    
    1. **Comprehensive EDA**: Analyzed 215,029+ repositories with 15+ different analyses
    2. **Data Preprocessing**: 
       - Handled missing values in Language and License columns
       - Removed multicollinearity (Watchers vs Stars)
       - Applied log transformations to handle extreme skewness
       - Encoded categorical variables
       - Scaled features for optimal model performance
    
    3. **Model Development**: 
       - Trained 6 different ML models
       - Compared performance using R², RMSE, and MAE metrics
       - Selected best performing model for deployment
    
    4. **Interactive Application**: 
       - Built user-friendly Streamlit interface
       - Enabled runtime predictions
       - Visualized insights effectively
    """)
    
    # Key Insights
    st.markdown("### 💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Data Insights:**
        - 📊 Strong correlation between Forks and Stars (r > 0.75)
        - 🌐 Python is the most popular language
        - 📈 Extreme right skewness in all numerical features
        - ⚠️ Perfect multicollinearity: Watchers ≡ Stars
        """)
    
    with col2:
        st.markdown("""
        **Model Insights:**
        - 🏆 Tree-based models outperformed linear models
        - 🎯 Forks is the strongest predictor of Stars
        - 📉 Log transformation crucial for handling skewness
        - ✅ Successfully deployed for real-time predictions
        """)
    
    # Future Improvements
    st.markdown("### 🚀 Future Improvements")
    
    improvements = """
    - 🔄 **Model Enhancement**: Hyperparameter tuning and ensemble methods
    - 📊 **Feature Engineering**: Create interaction features and temporal features
    - 🌐 **API Integration**: Connect to GitHub API for real-time data
    - 📱 **Mobile Support**: Responsive design for mobile devices
    - 🔍 **Advanced Analytics**: Sentiment analysis of repository descriptions
    - 📈 **Time Series**: Predict future star growth trends
    """
    st.markdown(improvements)
    
    # Metrics summary
    if models and 'metadata' in models:
        st.markdown("### 📊 Final Model Performance")
        metadata = models['metadata']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", metadata['model_name'])
        with col2:
            st.metric("Test R² Score", f"{metadata['test_r2']:.4f}")
        with col3:
            st.metric("Test RMSE", f"{metadata['test_rmse']:.4f}")
    
    # Thank you
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <h2>Thank You! 🎉</h2>
        <p style='font-size: 1.2rem; color: #555;'>
            Project completed as part of Introduction to Data Science (IDS F24)
        </p>
        <p style='color: #888;'>
            Author: Zohaib Khan | Course Instructor: Dr. M Nadeem Majeed
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
