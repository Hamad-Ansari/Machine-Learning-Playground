
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def regression_report(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return f"MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}"

# App title
st.title("Machine Learning Playground")
st.write("Upload your dataset and explore different ML models")

## 1. Data Upload Section
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "json"])

if uploaded_file is not None:
    # Read data
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.json'):
        df = pd.read_json(uploaded_file)
    
    # Show dataset
    st.subheader("Dataset Preview")
    st.write(df.head())
    
    ## 2. Data Exploration
    st.header("Data Exploration")
    selected_col = st.selectbox("Select a column to visualize", df.columns)
    
    fig, ax = plt.subplots()
    if pd.api.types.is_numeric_dtype(df[selected_col]):
        sns.histplot(df[selected_col], kde=True, ax=ax)
    else:
        df[selected_col].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)

    ## 3. Machine Learning Options
    st.header("Machine Learning Options")
    problem_type = st.radio("Select problem type", ["Classification", "Regression"])
    features = st.multiselect("Select features", df.columns)
    target = st.selectbox("Select target variable", df.columns)
    
    if features and target:
        X = df[features]
        y = df[target]
        
        # Preprocessing
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        
        # Model selection
        model_type = st.selectbox("Select model", ["Random Forest", "SVM", "Logistic Regression"] if problem_type == "Classification" else ["Random Forest", "Linear Regression", "SVR"])
        
        if problem_type == "Classification":
            if model_type == "Random Forest":
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier()
            elif model_type == "SVM":
                from sklearn.svm import SVC
                model = SVC()
            elif model_type == "Logistic Regression":
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression()
            
            # Encode target for classification
            le = LabelEncoder()
            y = le.fit_transform(y)
        else:
            if model_type == "Random Forest":
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor()
            elif model_type == "Linear Regression":
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
            elif model_type == "SVR":
                from sklearn.svm import SVR
                model = SVR()
        
        # Create pipeline
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train/test split
        test_size = st.slider("Test size ratio", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train model
        with st.spinner('Training model...'):
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                st.success("Model trained successfully!")
                st.subheader("Model Performance")
                
                if problem_type == "Classification":
                    st.text(classification_report(y_test, y_pred))
                else:
                    st.text(regression_report(y_test, y_pred))
                    
                    fig, ax = plt.subplots()
                    ax.scatter(y_test, y_pred)
                    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
                    ax.set_xlabel('Actual')
                    ax.set_ylabel('Predicted')
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Error during model training: {str(e)}")