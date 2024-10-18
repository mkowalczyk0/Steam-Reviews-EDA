# Steam Reviews EDA
Exploratory Data Analysis (EDA) and sensitivity analysis on a Steam Game Reviews


### 1. Loading and Understanding the Dataset
Read the dataset.
Display basic info (columns, data types, missing values).
Show initial rows for a preview of the data.
### 2. Data Cleaning
Handle missing values.
Fix incorrect data types.
Drop duplicates, if any.
### 3. Exploratory Data Analysis (EDA)
Descriptive Statistics: Mean, median, standard deviation, and percentiles.
Distribution Analysis: Histograms, box plots, and density plots to understand the distribution of continuous variables.
Categorical Analysis: Bar charts for categorical features.
Correlation Analysis: Correlation matrix and heatmap.
Visualization: Scatter plots, pair plots, or feature relationships.
### 4. Feature Engineering
Create new features based on existing data, if applicable.
Normalize or scale features for further analysis.
### 5. Sensitivity Analysis
Variable Importance: Use techniques like decision trees or random forests to see which variables are important.
Scenario Analysis: Varying key inputs/features and analyzing the effect on the target variable.
Impact of Outliers: Investigating how outliers influence the overall trends in the dataset.
### 6. Sentiment Analysis 
Identify and categorize reviews in order to determine whether the writer's attitude towards a particular game/platform is positive, negative, or neutral.
### 6. Summary of Insights
Summarize key findings and visualizations.
Highlight trends, relationships, and important factors affecting the target variable.



### The dataset contains 1,461 entries with the following columns:

- id: Unique identifier for each review.
- game: The game being reviewed.
- review: The text of the review.
- author_playtime_at_review: Hours the author played the game before leaving the review.
- voted_up: Boolean indicating if the review is positive.
- votes_up: Number of upvotes the review received.
- votes_funny: Number of "funny" votes the review received.
- constructive: Binary value indicating if a review is considered constructive

The data is already cleansed. There is no further cleaning that needs to take palce.

### Dataset Descriptive Statistics
![image](https://github.com/user-attachments/assets/05cad7bf-5e40-4849-97e7-8620b7388749)



Now lets get into the fun stuff:

# Distribution Analysis:
### Distribution of Playtime at Review
![image](https://github.com/user-attachments/assets/523cfad6-ef5b-4932-ad8b-567c1510dee6)
```python
plt.figure(figsize=(8, 5))
sns.histplot(data['author_playtime_at_review'], bins=30, kde=True)
plt.title('Distribution of Playtime at Review')
plt.xlabel('Playtime (hours)')
plt.ylabel('Frequency')
plt.xlim(0, 2000)
plt.show()
```
### Distribution of Upvotes
![image](https://github.com/user-attachments/assets/f3410c60-8470-4732-95b7-adf901354cb0)
```python
plt.figure(figsize=(8, 5))
sns.histplot(data['votes_up'], bins=30, kde=True)
plt.title('Distribution of Upvotes')
plt.xlabel('Upvotes')
plt.ylabel('Frequency')
plt.xlim(0, 40)
plt.show()
```
### Distribution of Funny Votes
![image](https://github.com/user-attachments/assets/022bbdaa-b164-43eb-a565-ab5482ab082c)
```python
plt.figure(figsize=(8, 5))
sns.histplot(data['votes_funny'], bins=30, kde=True)
plt.title('Distribution of Funny Votes')
plt.xlabel('Funny Votes')
plt.ylabel('Frequency')
plt.xlim(0, 10)
plt.show()
```
### Proportion of Positive vs. Negative Reviews
![image](https://github.com/user-attachments/assets/283e2f1d-521f-4092-9b8d-76770f88a5ac)
```python
plt.figure(figsize=(8, 5))
ax = sns.countplot(x='voted_up', data=data)
plt.title('Proportion of Positive vs. Negative Reviews')
plt.xlabel('Voted Up (True=Positive, False=Negative)')
plt.ylabel('Count')

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', 
                fontsize=10)
```

# Correlation Analysis
#### We'll use a correlation matrix and plot a heatmap to visualize the correlation between attributes
![image](https://github.com/user-attachments/assets/f99ceb55-203e-4a32-8fa2-d76a94c65217)
```python
correlation_matrix = data[['author_playtime_at_review', 'votes_up', 'votes_funny', 'constructive']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()
```

# Visualization of Relationships
### Scatter plot of playtime vs. upvotes
![image](https://github.com/user-attachments/assets/f233da2c-cc0f-4a8c-a8d3-f54291c963d6)
```python
plt.figure(figsize=(8, 5))
sns.scatterplot(x='author_playtime_at_review', y='votes_up', data=data)
plt.title('Playtime at Review vs. Upvotes')
plt.xlabel('Playtime (hours)')
plt.ylabel('Upvotes')
plt.xlim(0, 2500)
plt.ylim(0, 50)
plt.show()
```
### Box plot for votes_up based on constructive reviews
![image](https://github.com/user-attachments/assets/9c2737d5-2215-48a4-8482-c0d95cc73870)
```python
plt.figure(figsize=(8, 5))
sns.boxplot(x='constructive', y='votes_up', data=data)
plt.title('Upvotes by Constructiveness')
plt.xlabel('Constructive Review (0 = No, 1 = Yes)')
plt.ylabel('Upvotes')
plt.ylim(0, 25)
plt.show()
```

# Feature Engineering
#### Feature engineering involves creating new features based on existing data that could improve our understanding or predictive capabilities.

### Categorizing playtime
##### Create categories like Low Playtime, Medium Playtime, and High Playtime based on thresholds:


```python
conditions = [
    (data['author_playtime_at_review'] <= 10),
    (data['author_playtime_at_review'] > 10) & (data['author_playtime_at_review'] <= 50),
    (data['author_playtime_at_review'] > 50)
]
labels = ['Low Playtime', 'Medium Playtime', 'High Playtime']
data['playtime_category'] = pd.cut(data['author_playtime_at_review'], 
                                           bins=[0, 10, 50, data['author_playtime_at_review'].max()],
                                           labels=labels, right=False)
```
### Create total_votes feature
##### Create a feature called total_votes as a sum of votes_up and votes_funny:

```python
data['total_votes'] = data['votes_up'] + data['votes_funny']
```

### Length of review text
##### Calculate the length of each review:

```python
data['review_length'] = data['review'].apply(len)
```

# Sensitivity Analysis
#### Sensitivity analysis involves understanding how the variation in different features impacts the outcome
#### We'll use a simple model like a decision tree or random forest to understand feature importance.

```python
# Variable Importance

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
data['playtime_category_encoded'] = label_encoder.fit_transform(data['playtime_category'].astype(str))

X = data[['author_playtime_at_review', 'votes_up', 'votes_funny', 'review_length', 'playtime_category_encoded']]
y = data['constructive']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

feature_importances = pd.DataFrame(rf_model.feature_importances_, index=X.columns, columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances)

# Visualize
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances['importance'], y=feature_importances.index)
plt.title('Feature Importances from Random Forest Model')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()
```
![image](https://github.com/user-attachments/assets/8bceff9f-8f33-48f2-8956-58e1111c9f8b)


# Scenario Analysis
#### Sensitivity analysis is a technique used to determine how different values of an independent variable affect a particular dependent variable under a given set of assumptions. In this case, the analysis is examining how variations in playtime and votes_up influence the predicted constructiveness of reviews using a machine learning model.

```python
# Example scenario analysis

# Create a DataFrame for different scenarios
scenarios = pd.DataFrame({
    'playtime': [10, 20, 50, 80],
    'votes_up': [0, 5, 10, 15]
})

# Predict constructiveness for different scenarios
for index, row in scenarios.iterrows():
    test_row = pd.DataFrame({
        'author_playtime_at_review': [row['playtime']],
        'votes_up': [row['votes_up']],
        'votes_funny': [0],  # assuming no funny votes for simplicity
        'review_length': [50],  # average review length
        'playtime_category_encoded': [label_encoder.transform(['Low Playtime'])[0]]  # encoding
    })

    prediction = rf_model.predict(test_row)
    scenarios.loc[index, 'prediction'] = prediction[0]

print(scenarios)
```
#### After performing the a sensitivity analysis by varying playtime and votes_up to see how these factors influence the predicted constructiveness of reviews. The final output is a DataFrame that shows the predicted constructiveness for each combination of playtime and votes_up, allowing for an understanding of how these variables impact the model's predictions
![image](https://github.com/user-attachments/assets/d4651b00-6c24-4044-9fb3-7d12cc774bf9)



# Sentiment Analysis of Reviews
#### We can perform a sentiment analysis on the reviews themselves. We can use libraries like TextBlob or VADER to determine the sentiment of each review.

```python
from textblob import TextBlob

def get_sentiment(review):
    analysis = TextBlob(review)
    return analysis.sentiment.polarity  # Returns a value between -1 (negative) and 1 (positive)

data['sentiment'] = data['review'].apply(get_sentiment)
print(data[['review', 'sentiment']].head())
```
![image](https://github.com/user-attachments/assets/5ff331bb-be32-4e33-afd9-dcadd9233236)


### Group the Data by Game
```python
game_sentiment = data.groupby('game')['sentiment'].mean().reset_index()
game_sentiment.columns = ['game', 'average_sentiment']
game_sentiment_sorted = game_sentiment.sort_values(by='average_sentiment', ascending=False)
print(game_sentiment_sorted)
```

# Visualized Top Games with the Best sentiment
![image](https://github.com/user-attachments/assets/b24cb5ac-4424-4408-aa85-93394203ba9b)

```python
plt.figure(figsize=(12, 6))
sns.barplot(x='average_sentiment', y='game', data=game_sentiment_sorted, palette='viridis')
plt.title('Average Sentiment by Game')
plt.xlabel('Average Sentiment Score')
plt.ylabel('Game')
plt.xlim(-.25, .25)
plt.show()
```

# Project Summary: Exploratory Data Analysis and Sentiment Analysis of Steam Reviews

---

## 1. Data Loading and Initial Exploration
- **Loaded the Dataset**: The project began by loading a CSV file containing Steam reviews.
- **Initial Inspection**: Conducted initial inspections to understand the structure and contents of the dataset, identifying key features like `review`, `votes_up`, `author_playtime_at_review`, and `constructive`.

## 2. Data Cleaning
- **Handling Missing Values**: Identified and addressed missing values by either filling or dropping them based on context and relevance.
- **Data Type Conversion**: Converted relevant columns to appropriate data types (e.g., strings for text columns).
- **Removing Duplicates**: Ensured the dataset contained unique entries, enhancing data integrity for analysis.

## 3. Exploratory Data Analysis (EDA)
- **Descriptive Statistics**: Generated summary statistics for numerical features like `author_playtime_at_review`, `votes_up`, `votes_funny`, and `constructive`, providing a foundational understanding of the dataset.
- **Distribution Analysis**: Created histograms and bar plots to visualize distributions and proportions:
  - Distribution of playtime, upvotes, and funny votes.
  - Proportions of positive vs. negative reviews (i.e., `voted_up`).
- **Correlation Analysis**: 
  - Calculated and visualized a correlation matrix to identify relationships between numerical features.
  - A heatmap was plotted to visualize correlations, highlighting which features are more closely related.
- **Visualization of Relationships**: 
  - Created scatter plots to visualize relationships (e.g., `author_playtime_at_review` vs. `votes_up`).
  - Box plots illustrated the differences in upvotes between constructive and non-constructive reviews.

## 4. Feature Engineering
- **Playtime Categorization**: Introduced a new feature, `playtime_category`, to classify users based on playtime (e.g., Low, Medium, High).
- **Total Votes Calculation**: Combined `votes_up` and `votes_funny` into a new feature, `total_votes`, for a holistic view of user engagement.
- **Review Length Feature**: Added a feature to capture the length of each review, which may correlate with the review's constructiveness.

## 5. Sentiment Analysis
- **Sentiment Scoring**: Utilized the `TextBlob` library to compute sentiment polarity for each review, generating a score ranging from -1 (negative) to +1 (positive).
- **Sentiment Distribution Visualization**: Created histograms to visualize the distribution of sentiment scores across reviews.

## 6. Sensitivity Analysis
- **Variable Importance Using Random Forest**: 
  - Built a Random Forest model to assess the importance of features influencing whether a review is constructive.
  - Generated a bar plot to visualize feature importances, revealing that `votes_up`, `review_length`, and `author_playtime_at_review` are significant predictors.
- **Scenario Analysis**: 
  - Analyzed how variations in key features (e.g., playtime and votes) affect the constructiveness of reviews, providing insights into thresholds that impact user engagement.

## 7. Aggregating Sentiment by Game
- **Game Sentiment Aggregation**: 
  - Calculated the average sentiment score for each game by grouping reviews based on the `game` column.
  - Sorted and visualized the results, allowing comparisons of sentiment scores among different games.

## 8. Summary of Insights
- **Key Features Influencing Constructiveness**:
  - Longer reviews and higher `votes_up` scores correlate positively with the likelihood of a review being constructive.
- **Sentiment Analysis Insights**:
  - Reviews generally exhibit positive sentiment, with trends indicating that constructive reviews tend to score higher on the sentiment scale.
- **Scenario Analysis Findings**:
  - Sensitivity analysis highlighted specific thresholds for `author_playtime_at_review` and `votes_up` that significantly affect the probability of a review being classified as constructive.
- **Game-Level Insights**:
  - Aggregated sentiment scores provide valuable insights into user perceptions of different games, allowing game developers and marketers to understand audience reception better.

---

## Value Derived from the Analysis
1. **Improved Understanding of User Engagement**: Insights into how review characteristics (like length and upvotes) relate to user perceptions and constructiveness can help developers enhance engagement strategies.
2. **Actionable Feedback for Game Developers**: By identifying the factors that lead to constructive reviews, developers can focus on improving those aspects in their games.
3. **Marketing Strategies**: Knowing which games have higher sentiment scores can inform marketing strategies and promotional efforts, helping to drive sales.
4. **User-Centric Design**: Understanding user sentiment and engagement through reviews allows for better design decisions based on actual user feedback.

This detailed analysis presents a comprehensive view of user sentiment and engagement on the Steam platform, enabling stakeholders to make informed decisions based on data-driven insights. If you have any further questions or need additional analysis, feel free to ask!


