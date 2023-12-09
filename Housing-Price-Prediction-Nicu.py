import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Display basic information about the dataset
df = pd.read_csv('C:\\Things\\Housing.csv')

df.head(10)

df.info()

df.describe()

df.shape

df.isna().sum()

df.duplicated().sum()


# Renaming 'area' column to 'area(m2)'
df.rename(columns={'area':'area(m2)'},inplace=True)
df


# Separating numeric and categorical columns
numeric_cols=[]
cat_cols=[]
price_col=df.price
for column in df.columns:
    if pd.api.types.is_numeric_dtype(df[column]):
        print(f"The column '{column}' is numeric.")
        numeric_cols.append(column)
    else:
        print(f"The column '{column}' is not numeric.")
        cat_cols.append(column)


# Data Visualization: Univariate analysis for numeric columns
def univariate_analysis_numeric(col):
    fig, ax = plt.subplots(1, 3, figsize=(10,5))    
    sns.histplot(df[col], kde=True, bins=20, color='skyblue',ax=ax[0])
    ax[0].set_title(f'Histogram of {col} .')
    
    sns.boxplot(x=df[col],ax=ax[1])
    ax[1].set_title(f'Boxplot diagram of {col}')
    
    sns.violinplot(x=df[col],ax=ax[2])
    ax[2].set_title(f'Violinplot diagram of {col}')
    plt.show()
    
for col in numeric_cols:
    print(f' Univariate analysis for {col} column:')
    univariate_analysis_numeric(col)
        

# Data Visualization: Univariate analysis for categorical columns
def univariate_analysis_cat(col):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Countplot
    sns.countplot(data=df,x=df[col], palette='viridis', ax=ax[0])
    ax[0].set_title(f'Countplot for {col}')
    
    # Pie plot
    data_counts = df[col].value_counts()
    ax[1].pie(data_counts, labels=data_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
    ax[1].set_title(f'Pie plot for {col}')
    
    plt.show()

for col in cat_cols:
    print(f' Univariate analysis for {col} column:')
    univariate_analysis_cat(col)


# Data Visualization: Bivariate analysis between numeric columns and target variable 'price'
def bivariate_num_col(col):
     # Set the color palette
    sns.set_palette("tab10")
    
    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    # reg plot
    sns.regplot(x=df[col], y=df['price'], ax=ax[0])
    ax[0].set_title(f'Regression Plot: {col} vs Price')

    # Scatter plot
    sns.scatterplot(x=col, y='price', data=df, ax=ax[1])
    ax[1].set_title(f'Scatter Plot of {col} vs price')
    

    plt.tight_layout()
    plt.show()
    
    
for col in numeric_cols[1:]:
    print(f'Bivariate analysis between {col} and price')
    bivariate_num_col(col)


# Data Visualization: Bivariate analysis between categorical columns and 'price'
def bivariate_cat(col):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    sns.barplot(x=col, y='price', data=df, ax=ax[0])
    ax[0].set_title(f'Average Price by {col}')

    sns.boxplot(x=col, y='price', data=df, ax=ax[1])
    ax[1].set_title(f'Price Distribution by {col}')
    plt.tight_layout()
    plt.show()
    
for col in cat_cols[1:]:
    print(f'Bivariate analysis between {col} and price')
    bivariate_cat(col)


# Encoding categorical columns using LabelEncoder
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

for col in cat_cols:
    df[col]=le.fit_transform(df[col])
df


# Splitting data into features (X) and target variable (y)
X = df.drop('price', axis=1)
y = df['price']


# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)


# Model evaluation using metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
print("mean squared error:" , mean_squared_error(y_test,y_pred), "\n")
print("r2_score : ", r2_score(y_test, y_pred))


# Conclusion:
# 
# The analysis revealed that 'area(m2)', 'number_of_rooms', and 'location' significantly impact housing prices, showing strong correlations with property values. Larger property sizes and specific locations tend to command higher prices. The Linear Regression model provided reasonable predictive capability, explaining a notable percentage of the variance in housing prices. However, anomalies in larger property areas impacted predictive accuracy. Further outlier handling and advanced modeling techniques could enhance the model's performance.
# 
# This analysis offers crucial insights into housing price determinants, aiding stakeholders in real estate decision-making and suggesting avenues for refining predictive models.
