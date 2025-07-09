# 🎯 Comment Categorization Tool

A powerful Streamlit application for analyzing and categorizing user comments using AI-powered classification. This tool helps content creators, community managers, and social media analysts automatically categorize comments into meaningful categories like praise, support, criticism, abuse, threats, and more.

## 🚀 Features

- **AI-Powered Classification**: Uses machine learning models (Logistic Regression, SVM, BERT & distilbert) to categorize comments
- **8 Pre-defined Categories**: Praise, Support, Constructive Criticism, Hate/Abuse, Threats, Emotional, Spam, Questions/Suggestions
- **Multiple Input Methods**: Text input, CSV file upload, or sample data
- **Custom Training Data**: Upload your own training data or use the built-in sample dataset
- **Interactive Visualizations**: Charts and graphs showing category distribution and confidence scores
- **Export Functionality**: Download results as CSV, JSON, or comprehensive reports
- **Real-time Analysis**: Instant comment classification with confidence scores
- **Rule-based Fallback**: Ensures classification even without trained models

## 📋 Table of Contents

- Installation
- Quick Start
- Usage
- Categories
- Training Data
- Model Types
- Input Methods
- Output and Export
- API Reference

## 🛠️ Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/jatiwalnikhil61/comment-categorisation-tool.git
cd comment-categorization-tool
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download NLTK Data

The application will automatically download required NLTK data on first run, but you can also do it manually:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## 🚀 Quick Start

1. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

2. **Open in Browser**: Navigate to `http://localhost:8501`

3. **Train a Model**: 
   - Use the default sample data or upload your own training data
   - Select a model type (for e.g. Logistic Regression or SVM)
   - Click "Train Model"

4. **Analyze Comments**:
   - Enter comments in the text area or upload a CSV file
   - Click "Analyze Comments"
   - View results and export data

## 📖 Usage

### Basic Workflow

1. **Configuration** (Sidebar):
   - Choose training data source
   - Select model type
   - Train the model

2. **Input Comments** (Left Column):
   - Choose input method
   - Enter or upload comments
   - Click analyze

3. **View Results** (Right Column):
   - Summary metrics
   - Category distribution charts
   - Detailed results with confidence scores
   - Export options

### Training Data Options

#### 1. Default Sample Data
- 120 pre-labeled comments across 8 categories
- Balanced dataset for immediate use
- Good for testing and demonstration

#### 2. Upload CSV File
- Format: CSV with comment and category columns
- Supports any number of samples
- Automatic column detection

#### 3. Manual Input
- Format: `comment|category` (one per line)
- Useful for small datasets
- Real-time input and training

## 🏷️ Categories

The tool classifies comments into 8 predefined categories:

| Category | Description | Color | Example |
|----------|-------------|-------|---------|
| **Praise** | Positive feedback and compliments | Green | "Amazing work! Loved the animation." |
| **Support** | Encouraging and motivational comments | Blue | "Keep going, you're doing great!" |
| **Constructive Criticism** | Helpful feedback for improvement | Yellow | "Good content but could use better lighting." |
| **Hate/Abuse** | Negative, hostile, or abusive comments | Red | "This is trash, quit now." |
| **Threat** | Threatening or intimidating language | Purple | "I'll report you if this continues." |
| **Emotional** | Personal, emotional responses | Orange | "This reminded me of my childhood." |
| **Spam** | Promotional or irrelevant content | Gray | "Follow me for followers." |
| **Question/Suggestion** | Inquiries and suggestions | Teal | "Can you make one on topic X?" |

## 🤖 Model Types

### Logistic Regression
- **Best for**: General-purpose classification
- **Advantages**: Fast training, interpretable results
- **Recommended for**: Most use cases

### Support Vector Machine (SVM)
- **Best for**: Complex text patterns
- **Advantages**: High accuracy for well-separated classes
- **Recommended for**: High-quality training data

### BERT
- **Best for**: Complex language understanding and context-dependent classification
- **Advantages**: Deep contextual understanding, high accuracy on nuanced text
- **Recommended for**: Tasks requiring semantic understanding, sentiment analysis, document classification with sufficient computational resources

### DistilBERT
- **Best for**: Fast inference with good performance on most NLP tasks
- **Advantages**: 60% smaller than BERT, 60% faster inference while retaining 97% of BERT's performance
- **Recommended for**: Production environments with latency constraints, mobile/edge deployment, cost-sensitive applications

### Text Processing Pipeline
1. **Preprocessing**: Lowercase, remove special characters
2. **Tokenization**: Split into individual words
3. **Stopword Removal**: Remove common words
4. **Lemmatization**: Reduce words to root forms
5. **Vectorization**: Convert to numerical features (TF-IDF)
6. **Classification**: Predict category and confidence

## 📊 Input Methods

### 1. Text Input
- **Format**: One comment per line
- **Best for**: Quick analysis of small batches
- **Example**:
  ```
  Amazing work!
  This could be better.
  What software do you use?
  ```

### 2. File Upload
- **Format**: CSV files with comment column
- **Best for**: Large datasets
- **Requirements**: Headers in first row

### 3. Sample Data
- **Format**: Built-in examples
- **Best for**: Testing and demonstration
- **Size**: 10 sample comments from training data

## 📈 Output and Export

### Visualizations
- **Summary Metrics**: Total comments, categories found, confidence scores
- **Pie Chart**: Category distribution
- **Bar Chart**: Comment count by category
- **Histogram**: Confidence score distribution

### Detailed Results
- **Filtering**: By category and confidence threshold
- **Card View**: Individual comment cards with category labels
- **Color Coding**: Visual category identification

### Export Options
- **CSV**: Structured data for further analysis
- **JSON**: API-friendly format
- **Report**: Comprehensive markdown report with insights

## 🔧 API Reference

### CommentCategorizer Class

#### Methods

```python
# Initialize the categorizer
categorizer = CommentCategorizer()

# Train a model
model, report = categorizer.train_model(model_type='logistic')

# Classify a single comment
category, confidence = categorizer.classify_comment("Great work!")

# Preprocess text
processed = categorizer.preprocess_text("This is a sample comment!")
```

#### Properties

```python
# Available categories
categorizer.categories

# Text preprocessing components
categorizer.lemmatizer
categorizer.stop_words
```

### Training Data Format

#### CSV Format
```csv
comment,category
"Amazing work!",praise
"This needs improvement",constructive_criticism
"What software do you use?",question_suggestion
```

#### Manual Input Format
```
Amazing work!|praise
This needs improvement|constructive_criticism
What software do you use?|question_suggestion
```

## 🎯 Performance Tips

### For Better Accuracy
1. **Use Balanced Training Data**: Include similar numbers of examples for each category
2. **Quality over Quantity**: Well-labeled examples are better than many poor examples
3. **Domain-Specific Training**: Train on comments similar to your target domain
4. **Regular Retraining**: Update models with new examples periodically

### For Better Performance
1. **Batch Processing**: Analyze multiple comments at once
2. **Confidence Thresholds**: Use confidence scores to filter results
3. **Rule-based Fallback**: Leverage built-in rules for edge cases

## 🔍 Troubleshooting

### Common Issues

#### Model Training Fails
- **Cause**: Insufficient or imbalanced training data
- **Solution**: Add more examples, ensure all categories are represented

#### Low Confidence Scores
- **Cause**: Comments differ significantly from training data
- **Solution**: Add domain-specific training examples

#### NLTK Download Errors
- **Cause**: Network issues or permissions
- **Solution**: Manual download or run with administrator privileges

#### Memory Issues
- **Cause**: Large datasets or complex models
- **Solution**: Process in smaller batches, use simpler models

## 🚀 Advanced Usage

### Custom Categories
To add new categories, modify the `categories` dictionary in the `CommentCategorizer` class:

```python
self.categories = {
    'your_category': {'label': 'Your Category', 'color': '#your_color'},
    # ... existing categories
}
```

### Custom Preprocessing
Override the `preprocess_text` method for domain-specific preprocessing:

```python
def preprocess_text(self, text):
    # Your custom preprocessing logic
    return processed_text
```

## 📚 Dependencies

- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Plotting library
- **seaborn**: Statistical data visualization
- **plotly**: Interactive visualizations
- **scikit-learn**: Machine learning library
- **nltk**: Natural language processing
- **torch**: Deep learning framework (optional)
