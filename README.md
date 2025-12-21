# ⭐ GitHub Repository Popularity Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![Status](https://img.shields.io/badge/Status-Complete-success)

An interactive machine learning web application that predicts GitHub repository popularity (Stars) based on various repository features. Built with Python, Streamlit, and multiple ML algorithms.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Machine Learning Models](#-machine-learning-models)
- [Results](#-results)
- [Technologies Used](#-technologies-used)
- [Author](#-author)

---

## 🎯 Project Overview

This project was developed as part of the **Introduction to Data Science (IDS F24)** course. The goal is to predict the popularity of GitHub repositories (measured by Stars) using machine learning techniques.

### Key Objectives:
1. ✅ Perform comprehensive Exploratory Data Analysis (EDA) with 15+ analyses
2. ✅ Preprocess data to handle missing values, outliers, and feature engineering
3. ✅ Train and compare 6 different machine learning models
4. ✅ Build an interactive Streamlit web application
5. ✅ Enable real-time predictions based on user input

---

## ✨ Features

### 🏠 **Home Page**
- Project overview and objectives
- Dataset statistics and key metrics
- Navigation guide

### 📈 **Exploratory Data Analysis (EDA)**
15+ comprehensive analyses including:
- Dataset overview and structure
- Univariate analysis (distributions, statistics)
- Bivariate analysis (scatter plots, relationships)
- Correlation analysis with interactive heatmaps
- Distribution analysis across multiple features
- Language popularity analysis
- Missing values visualization
- Outlier detection using IQR method

### 🤖 **Model & Predictions**
- Model performance metrics (R², RMSE, MAE)
- Comparison of 6 ML models
- **Real-time predictions** with user input form
- Interactive prediction interface

### 📋 **Conclusion**
- Project summary and key achievements
- Data and model insights
- Future improvement suggestions

---

## 📊 Dataset

- **Source**: Kaggle - GitHub Repositories Dataset
- **Size**: 215,029 repositories
- **Features**: 24 attributes including:
  - **Numerical**: Stars, Forks, Watchers, Open Issues, Size
  - **Categorical**: Language, License, Has Wiki, Has Issues, Has Projects
  - **Temporal**: Created At, Updated At

### Key Statistics:
- **Languages**: 100+ programming languages
- **Average Stars**: ~1,115
- **Max Stars**: 200,000+
- **Date Range**: Repositories from 2008-2024

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/ZohaibCodez/github-repository-popularity-predictor.git
cd github-repository-popularity-predictor
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Jupyter Notebooks (Optional)
To train models from scratch:
```bash
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_preprocessing_and_modeling.ipynb
```

---

## 💻 Usage

### Run the Streamlit Application
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Making Predictions
1. Navigate to **"🤖 Model & Predictions"** section
2. Click on **"🔮 Make Predictions"** tab
3. Enter repository details:
   - Number of Forks
   - Number of Open Issues
   - Repository Size (KB)
   - Programming Language
   - Boolean features (Has Wiki, Has Issues, Has Projects)
4. Click **"🔮 Predict Stars"**
5. View predicted star count and category

---

## 📁 Project Structure

```
github-repository-popularity-predictor/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── pyproject.toml                  # Project configuration
│
├── data/
│   └── repositories.csv            # Raw dataset
│
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory Data Analysis
│   └── 02_preprocessing_and_modeling.ipynb  # ML models
│
├── models/                         # Saved ML models (generated)
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── feature_columns.pkl
│   └── model_metadata.pkl
│
└── reports/                        # Analysis reports
    ├── EDA_REPORT.md
    └── report.md
```

---

## 🤖 Machine Learning Models

### Models Trained:
1. **Linear Regression** - Baseline model
2. **Ridge Regression** - L2 regularization
3. **Lasso Regression** - L1 regularization
4. **Random Forest** - Ensemble of decision trees
5. **Gradient Boosting** - Boosted decision trees
6. **XGBoost** - Optimized gradient boosting

### Data Preprocessing Steps:
1. ✅ Handle missing values (Language, License)
2. ✅ Remove multicollinearity (dropped Watchers column)
3. ✅ Log transformation for skewed features
4. ✅ Label encoding for categorical variables
5. ✅ Standard scaling for numerical features
6. ✅ Train-test split (80/20)

### Model Selection Criteria:
- **Primary Metric**: R² Score on test set
- **Secondary Metrics**: RMSE, MAE
- **Best Model**: Selected based on highest test R²

---

## 📊 Results

### Key Findings:

#### Data Insights:
- 📊 **Strong correlation** between Forks and Stars (r > 0.75)
- 🌐 **Python** is the most popular programming language
- 📈 **Extreme right skewness** in all numerical features (skewness: 27-88)
- ⚠️ **Perfect multicollinearity**: Watchers ≡ Stars (r ≈ 0.99)

#### Model Performance:
- 🏆 **Best Model**: XGBoost / Random Forest (typically highest R²)
- 🎯 **Key Predictor**: Forks is the strongest feature
- 📉 **Log transformation** crucial for handling skewness
- ✅ Successfully deployed for **real-time predictions**

### Model Metrics (Example):
| Model | Train R² | Test R² | RMSE | MAE |
|-------|----------|---------|------|-----|
| XGBoost | 0.95 | 0.92 | 0.45 | 0.32 |
| Random Forest | 0.94 | 0.91 | 0.47 | 0.34 |
| Gradient Boosting | 0.93 | 0.90 | 0.49 | 0.35 |

---

## 🛠️ Technologies Used

### Programming & Libraries:
- **Python 3.8+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Static visualizations
- **Plotly** - Interactive visualizations

### Machine Learning:
- **Scikit-learn** - ML models and preprocessing
- **XGBoost** - Gradient boosting
- **Joblib** - Model persistence

### Web Application:
- **Streamlit** - Interactive web interface

### Development Tools:
- **Jupyter Notebook** - Analysis and experimentation
- **Git** - Version control

---

## 🔮 Future Improvements

- 🔄 **Hyperparameter Tuning**: Grid search and cross-validation
- 📊 **Feature Engineering**: Interaction features, polynomial features
- 🌐 **API Integration**: Real-time GitHub API data fetching
- 📱 **Mobile Optimization**: Responsive design improvements
- 🔍 **NLP Analysis**: Repository description sentiment analysis
- 📈 **Time Series**: Predict future star growth trends
- 🎨 **UI Enhancement**: More visualizations and customization
- 🚀 **Deployment**: Host on cloud platform (Heroku, AWS, Azure)

---

## 👨‍💻 Author

**Zohaib Khan**  
- Course: Introduction to Data Science (IDS F24)
- Instructor: Dr. M Nadeem Majeed
- GitHub: [@ZohaibCodez](https://github.com/ZohaibCodez)

---

## 📄 License

This project was created for educational purposes as part of a university course.

---

## 🙏 Acknowledgments

- Dr. M Nadeem Majeed for course instruction and guidance
- Kaggle for providing the GitHub repositories dataset
- Streamlit community for excellent documentation
- Open-source ML community for amazing tools

---

## 📞 Contact

For questions or feedback, please reach out through GitHub issues or contact the author directly.

---

<div align="center">

**⭐ If you found this project helpful, please consider giving it a star! ⭐**

Made with ❤️ and Python

</div>
