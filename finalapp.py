
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import time
import json
import io
import os
from datetime import datetime
import warnings
import torch
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

# Initialize session state
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'analyzed_comments' not in st.session_state:
    st.session_state.analyzed_comments = []
if 'training_data' not in st.session_state:
    st.session_state.training_data = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = 'logistic'

class CommentCategorizer:
    def __init__(self):
        self.categories = {
            'Praise': {'label': 'Praise', 'color': '#28a745'},
            'Support': {'label': 'Support', 'color': '#17a2b8'},
            'Constructive Criticism': {'label': 'Constructive Criticism', 'color': '#ffc107'},
            'Hate/Abuse': {'label': 'Hate/Abuse', 'color': '#dc3545'},
            'Threat': {'label': 'Threat', 'color': '#6f42c1'},
            'Emotional': {'label': 'Emotional', 'color': '#fd7e14'},
            'Irrelevant/Spam': {'label': 'Irrelevant/Spam', 'color': '#6c757d'},
            'Question/Suggestion': {'label': 'Question/Suggestion', 'color': '#20c997'}
        }
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Default sample dataset
        self.default_sample_data = self.create_default_sample_dataset()
        
    def create_default_sample_dataset(self):
        """Create a default labeled dataset for training"""
        data = [
            # Praise
            ("Amazing work! Loved the animation.", "Praise"),
            ("This is absolutely fantastic!", "Praise"),
            ("Outstanding content, keep it up!", "Praise"),
            ("Brilliant work, you're so talented!", "Praise"),
            ("Perfect! This is exactly what I needed.", "Praise"),
            ("Incredible job on this project!", "Praise"),
            ("Excellent quality, very impressed!", "Praise"),
            ("Wonderful work, thank you so much!", "Praise"),
            ("This is pure gold, amazing!", "Praise"),
            ("Spectacular work, love every bit of it!", "Praise"),
            ("Awesome content, you're the best!", "Praise"),
            ("Beautiful work, truly inspiring!", "Praise"),
            ("Fantastic job, exceeded my expectations!", "Praise"),
            ("Superb quality, keep up the great work!", "Praise"),
            ("Magnificent work, absolutely love it!", "Praise"),
            
            # Support
            ("Keep going, you're doing great!", "Support"),
            ("Don't give up, you've got this!", "Support"),
            ("We believe in you!", "Support"),
            ("Your hard work is paying off!", "Support"),
            ("Stay strong, you're making progress!", "Support"),
            ("Keep pushing forward!", "Support"),
            ("You're on the right track!", "Support"),
            ("Keep up the good work!", "Support"),
            ("We're rooting for you!", "Support"),
            ("You're doing amazing, don't stop!", "Support"),
            ("Keep at it, you're improving!", "Support"),
            ("Stay motivated, you're doing well!", "Support"),
            ("Keep going, we Support you!", "Support"),
            ("You're making great progress!", "Support"),
            ("Keep it up, you're doing fantastic!", "Support"),
            
            # Constructive Criticism
            ("The animation was okay but the voiceover felt off.", "Constructive Criticism"),
            ("Good content but could use better lighting.", "Constructive Criticism"),
            ("I liked the idea but the execution could be improved.", "Constructive Criticism"),
            ("Nice work, though the audio quality could be better.", "Constructive Criticism"),
            ("Good effort, but the pacing seemed a bit slow.", "Constructive Criticism"),
            ("The content is good but needs better organization.", "Constructive Criticism"),
            ("Decent work, but the transitions could be smoother.", "Constructive Criticism"),
            ("Good concept, but the delivery could be more engaging.", "Constructive Criticism"),
            ("Nice try, but the background music was distracting.", "Constructive Criticism"),
            ("Good work overall, but the ending felt rushed.", "Constructive Criticism"),
            ("The idea is great, but the presentation needs work.", "Constructive Criticism"),
            ("Good content, but the visuals could be clearer.", "Constructive Criticism"),
            ("Nice effort, but the structure could be better.", "Constructive Criticism"),
            ("Good work, though the tone could be more consistent.", "Constructive Criticism"),
            ("Decent content, but the length could be optimized.", "Constructive Criticism"),
            
            # Hate/Abuse
            ("This is trash, quit now.", "Hate/Abuse"),
            ("Worst content ever, you suck.", "Hate/Abuse"),
            ("This is garbage, complete waste of time.", "Hate/Abuse"),
            ("Terrible work, you're pathetic.", "Hate/Abuse"),
            ("This is awful, you have no talent.", "Hate/Abuse"),
            ("Stupid content, you're an idiot.", "Hate/Abuse"),
            ("This sucks, why do you even try?", "Hate/Abuse"),
            ("Horrible work, you're useless.", "Hate/Abuse"),
            ("This is the worst thing I've ever seen.", "Hate/Abuse"),
            ("Disgusting content, you should be ashamed.", "Hate/Abuse"),
            ("This is trash, complete failure.", "Hate/Abuse"),
            ("Terrible quality, you're a joke.", "Hate/Abuse"),
            ("This is garbage, stop making content.", "Hate/Abuse"),
            ("Awful work, you have no skills.", "Hate/Abuse"),
            ("This sucks so bad, give up already.", "Hate/Abuse"),
            
            # Threat
            ("I'll report you if this continues.", "Threat"),
            ("I'm going to sue you for this.", "Threat"),
            ("I'll make sure everyone knows how bad you are.", "Threat"),
            ("I'm going to get you shut down.", "Threat"),
            ("I'll destroy your reputation.", "Threat"),
            ("I'm reporting this to authorities.", "Threat"),
            ("I'll make sure you never work again.", "Threat"),
            ("I'm going to take legal action.", "Threat"),
            ("I'll ruin your career for this.", "Threat"),
            ("I'm going to expose you everywhere.", "Threat"),
            ("I'll make sure you pay for this.", "Threat"),
            ("I'm going to report you to your boss.", "Threat"),
            ("I'll get you banned from everywhere.", "Threat"),
            ("I'm going to make your life miserable.", "Threat"),
            ("I'll ensure you never succeed.", "Threat"),
            
            # Emotional
            ("This reminded me of my childhood.", "Emotional"),
            ("This made me cry, so touching.", "Emotional"),
            ("This brings back so many memories.", "Emotional"),
            ("I'm feeling so nostalgic right now.", "Emotional"),
            ("This touched my heart deeply.", "Emotional"),
            ("This made me think of my grandmother.", "Emotional"),
            ("I'm emotional watching this.", "Emotional"),
            ("This reminds me of better times.", "Emotional"),
            ("This brought tears to my eyes.", "Emotional"),
            ("I'm feeling so connected to this.", "Emotional"),
            ("This makes me feel so happy.", "Emotional"),
            ("I'm overwhelmed with emotion.", "Emotional"),
            ("This reminds me of my late father.", "Emotional"),
            ("This gives me chills every time.", "Emotional"),
            ("I'm feeling so inspired right now.", "Emotional"),
            
            # Irrelevant/Spam
            ("Follow me for followers.", "Irrelevant/Spam"),
            ("Check out my profile for amazing content!", "Irrelevant/Spam"),
            ("Subscribe to my channel!", "Irrelevant/Spam"),
            ("DM me for collaboration opportunities.", "Irrelevant/Spam"),
            ("Link in bio for exclusive content.", "Irrelevant/Spam"),
            ("Follow for follow back!", "Irrelevant/Spam"),
            ("Check out my latest video!", "Irrelevant/Spam"),
            ("Visit my website for more!", "Irrelevant/Spam"),
            ("Click here for free stuff!", "Irrelevant/Spam"),
            ("Follow me on all platforms!", "Irrelevant/Spam"),
            ("Check my bio for discount codes!", "Irrelevant/Spam"),
            ("Subscribe for daily updates!", "Irrelevant/Spam"),
            ("Follow me for lifestyle content!", "Irrelevant/Spam"),
            ("Check out my store in bio!", "Irrelevant/Spam"),
            ("DM me for business inquiries!", "Irrelevant/Spam"),
            
            # Question/Suggestion
            ("Can you make one on topic X?", "Question/Suggestion"),
            ("What software do you use for editing?", "Question/Suggestion"),
            ("Could you do a tutorial on this?", "Question/Suggestion"),
            ("How did you create this effect?", "Question/Suggestion"),
            ("Can you explain the process?", "Question/Suggestion"),
            ("What's your setup for recording?", "Question/Suggestion"),
            ("Could you make a behind-the-scenes video?", "Question/Suggestion"),
            ("What camera do you use?", "Question/Suggestion"),
            ("Can you share the source files?", "Question/Suggestion"),
            ("How long did this take to make?", "Question/Suggestion"),
            ("Could you do a collaboration?", "Question/Suggestion"),
            ("What's your creative process?", "Question/Suggestion"),
            ("Can you make more content like this?", "Question/Suggestion"),
            ("How do you come up with ideas?", "Question/Suggestion"),
            ("Could you review my work?", "Question/Suggestion"),
        ]
        
        df = pd.DataFrame(data, columns=['comment', 'category'])
        return df
    
    def preprocess_text(self, text):
        """Preprocess text for analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def get_tokenizer_and_model(self, model_type):
        """Return appropriate tokenizer and model based on model type"""
        if model_type == 'bert':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased', 
                num_labels=len(self.categories)
            )
        elif model_type == 'distilbert':
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            model = DistilBertForSequenceClassification.from_pretrained(
                'distilbert-base-uncased', 
                num_labels=len(self.categories)
            )
        else:
            return None, None
        return tokenizer, model

    def prepare_transformer_data(self, X, y, tokenizer, max_length=128):
        """Prepare data for transformer model training"""
        encodings = tokenizer(
            X.tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        # Convert labels to numerical indices
        label_map = {cat: idx for idx, cat in enumerate(self.categories.keys())}
        labels = torch.tensor([label_map[label] for label in y])
        dataset = TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask'],
            labels
        )
        return dataset, label_map
        
    def prepare_training_data(self):
        """Prepare training data from session state or default"""
        if st.session_state.training_data is not None:
            return st.session_state.training_data
        else:
            return self.default_sample_data
        
    def train_model(self, model_type='logistic'):
        """Train the classification model"""
        # Get training data
        training_data = self.prepare_training_data()
        
        if training_data.empty:
            st.error("No training data available. Please upload or input training data.")
            return None, None
        
        # Preprocess the data
        X = training_data['comment']  # Don't preprocess for transformers
        y = training_data['category']
        
        # Split data
        if len(training_data) < 10:
            st.warning("Training data is very small. Consider adding more samples for better performance.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

        st.session_state.model_type = model_type
        if model_type in ['bert', 'distilbert']:
            # Skip preprocessing for transformers
            return self.train_transformer_model(X_train, X_test, y_train, y_test, model_type)
        else:
            # Preprocess for traditional models
            X_train = X_train.apply(self.preprocess_text)
            X_test = X_test.apply(self.preprocess_text)
            return self.train_traditional_model(X_train, X_test, y_train, y_test, model_type)

    def train_traditional_model(self, X_train, X_test, y_train, y_test, model_type):
        """Train traditional ML models (Logistic or SVM)"""
        if model_type == 'logistic':
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ('classifier', LogisticRegression(random_state=42, max_iter=1000))
            ])
        else:  # SVM
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ('classifier', SVC(random_state=42, probability=True))
            ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        st.session_state.trained_model = pipeline
        return pipeline, classification_report(y_test, y_pred, output_dict=True)
    
    def train_transformer_model(self, X_train, X_test, y_train, y_test, model_type):
        """Train transformer-based model (BERT or DistilBERT)"""
        # Get tokenizer and model
        tokenizer, model = self.get_tokenizer_and_model(model_type)
        if tokenizer is None or model is None:
            st.error(f"Invalid model type: {model_type}")
            return None, None

        # Move model to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Prepare datasets
        train_dataset, label_map = self.prepare_transformer_data(X_train, y_train, tokenizer)
        test_dataset, _ = self.prepare_transformer_data(X_test, y_test, tokenizer)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16)

        # Training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        num_epochs = 3  # Adjust based on needs

        # Training loop
        model.train()
        for epoch in range(num_epochs):
            for batch in train_loader:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        # Evaluation
        model.eval()
        predictions, true_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        # Convert numerical predictions back to category labels
        reverse_label_map = {idx: cat for cat, idx in label_map.items()}
        y_pred = [reverse_label_map[pred] for pred in predictions]
        y_true = [reverse_label_map[label] for label in true_labels]

        # Store model and related objects
        pipeline = {
            'model': model,
            'tokenizer': tokenizer,
            'label_map': label_map,
            'device': device
        }
        st.session_state.trained_model = pipeline
        return pipeline, classification_report(y_true, y_pred, output_dict=True)


    def classify_comment(self, comment):
        """Classify a single comment"""
        if st.session_state.get("trained_model") is None:
            return self.rule_based_classification(comment)

        try:
            model_type = st.session_state.model_type
            if model_type in ['bert', 'distilbert']:
                pipeline = st.session_state.trained_model
                model = pipeline['model']
                tokenizer = pipeline['tokenizer']
                label_map = pipeline['label_map']
                device = pipeline['device']

                # Tokenize input
                encodings = tokenizer(
                    [comment],
                    truncation=True,
                    padding=True,
                    max_length=128,
                    return_tensors='pt'
                )
                input_ids = encodings['input_ids'].to(device)
                attention_mask = encodings['attention_mask'].to(device)

                # Predict
                model.eval()
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask)
                    probs = torch.softmax(outputs.logits, dim=1)
                    confidence, pred_idx = probs.max(dim=1)
                    prediction = list(label_map.keys())[pred_idx.item()]
                    confidence = confidence.item()
                return prediction, confidence
            else:
                processed_comment = self.preprocess_text(comment)
                model = st.session_state.trained_model
                prediction = model.predict([processed_comment])[0]
                confidence = model.predict_proba([processed_comment]).max()
                return prediction, confidence
        except Exception as e:
            st.error(f"Error in classification: {str(e)}")
            return self.rule_based_classification(comment)


    def rule_based_classification(self, comment):
        """Rule-based fallback classifier"""
        text = comment.lower()
        patterns = {
            'Praise': r'\b(amazing|awesome|fantastic|great|excellent|perfect|love|loved|brilliant|outstanding|incredible|wonderful|beautiful|superb|magnificent)\b',
            'Support': r'\b(keep going|Support|encourage|you can do it|don\'t give up|proud of you|rooting for you|believe in you|stay strong|keep up)\b',
            'Constructive Criticism': r'\b(but|however|though|could|might|suggest|improve|better|feedback|okay but|good but|nice but)\b',
            'Hate/Abuse': r'\b(trash|hate|terrible|awful|worst|stupid|idiot|sucks|garbage|pathetic|useless|quit|failure|disgusting)\b',
            'Threat': r'\b(report|sue|legal action|shut down|destroy|ruin|expose|authorities|banned|take down|make sure you)\b',
            'Emotional': r'\b(reminds me|childhood|memories|feel|feeling|Emotional|tears|nostalgic|brings back|touched|heart|cry)\b',
            'Irrelevant/Spam': r'\b(follow me|subscribe|check out|my channel|my profile|dm me|click here|link in bio|follow for follow|visit my)\b',
            'Question/Suggestion': r'\b(can you|could you|how do|what|when|where|tutorial|guide|suggestion|\?)\b'
        }

        for category, pattern in patterns.items():
            if re.search(pattern, text):
                return category, 0.8

        return 'Support', 0.5  # default fallback

def main():
    st.set_page_config(
        page_title="Comment Categorization Tool",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .category-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .comment-text {
        font-style: italic;
        color: #495057;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Comment Categorization Tool</h1>
        <p>Analyze and categorize user comments with AI-powered classification</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize categorizer
    categorizer = CommentCategorizer()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Training Data Section
        st.subheader("📚 Training Data")
        
        training_data_method = st.radio(
            "Choose training data source:",
            ["Use Default Sample Data", "Upload CSV File", "Manual Input"]
        )
        
        if training_data_method == "Upload CSV File":
            uploaded_training_file = st.file_uploader(
                "Upload training data CSV", 
                type=['csv'],
                key="training_file"
            )
            
            if uploaded_training_file is not None:
                try:
                    training_df = pd.read_csv(uploaded_training_file)
                    st.write("Preview of training data:")
                    st.dataframe(training_df.head())
                    
                    # Select columns
                    comment_col = st.selectbox("Select comment column:", training_df.columns)
                    category_col = st.selectbox("Select category column:", training_df.columns)
                    
                    if st.button("💾 Load Training Data"):
                        # Validate and load data
                        if comment_col and category_col:
                            training_data = training_df[[comment_col, category_col]].copy()
                            training_data.columns = ['comment', 'category']
                            training_data = training_data.dropna()
                            
                            st.session_state.training_data = training_data
                            st.success(f"✅ Loaded {len(training_data)} training samples!")
                            
                            # Show category distribution
                            st.write("Category distribution:")
                            category_counts = training_data['category'].value_counts()
                            st.bar_chart(category_counts)
                        else:
                            st.error("Please select both comment and category columns.")
                            
                except Exception as e:
                    st.error(f"Error loading training data: {str(e)}")
        
        elif training_data_method == "Manual Input":
            st.write("Enter training data (format: comment|category)")
            manual_training_input = st.text_area(
                "Training data:",
                height=200,
                placeholder="Amazing work!|Praise\nThis needs improvement|Constructive Criticism\nWhat software do you use?|Question/Suggestion"
            )
            
            if st.button("💾 Load Manual Training Data"):
                if manual_training_input:
                    try:
                        lines = [line.strip() for line in manual_training_input.split('\n') if line.strip()]
                        training_data = []
                        
                        for line in lines:
                            if '|' in line:
                                parts = line.split('|')
                                if len(parts) == 2:
                                    comment, category = parts[0].strip(), parts[1].strip()
                                    training_data.append({'comment': comment, 'category': category})
                        
                        if training_data:
                            training_df = pd.DataFrame(training_data)
                            st.session_state.training_data = training_df
                            st.success(f"✅ Loaded {len(training_data)} training samples!")
                            
                            # Show category distribution
                            st.write("Category distribution:")
                            category_counts = training_df['category'].value_counts()
                            st.bar_chart(category_counts)
                        else:
                            st.error("No valid training data found. Please check the format.")
                    except Exception as e:
                        st.error(f"Error processing manual input: {str(e)}")
                else:
                    st.warning("Please enter some training data.")
        
        else:  # Use Default Sample Data
            st.info("Using default sample dataset with 120 labeled comments.")
            if st.button("🔄 Reset to Default Data"):
                st.session_state.training_data = None
                st.success("✅ Reset to default training data!")
        
        st.divider()
        
        # Model Configuration
        st.subheader("🔧 Model Settings")
        model_type = st.selectbox(
            "Choose Model:", 
            ["logistic", "svm", "bert", "distilbert"],
            help="Traditional ML models (Logistic Regression, SVM) or Transformer models (BERT, DistilBERT)"
        )
        
        if st.button("🎯 Train Model"):
            with st.spinner(f"Training {model_type.upper()} model..."):
                model, report = categorizer.train_model(model_type)
                if model and report:
                    st.session_state.model_performance = report  # Store report
                    st.success("✅ Model trained successfully!")
                    
                    # Show training metrics
                    st.subheader("📊 Training Metrics")
                    accuracy = report['accuracy']
                    st.metric("Overall Accuracy", f"{accuracy:.2%}")
        
        st.divider()
        
        # Dataset Info
        st.subheader("📋 Dataset Information")
        training_data = categorizer.prepare_training_data()
        # st.info(f"Training samples: {len(training_data)}")
        # st.info(f"Categories: {len(categorizer.categories)}")
        
        # Category distribution
        if not training_data.empty:
            category_counts = training_data['category'].value_counts()
            st.bar_chart(category_counts)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input Comments")
        
        # Input methods
        input_method = st.radio("Choose input method:", ["Text Input", "File Upload", "Sample Data"])
        
        if input_method == "Text Input":
            comments_input = st.text_area(
                "Enter comments (one per line):",
                height=300,
                placeholder="Enter your comments here...\nExample:\nAmazing work!\nThis could be better.\nWhat software do you use?"
            )
            
            if st.button("🔍 Analyze Comments"):
                if comments_input:
                    comments = [c.strip() for c in comments_input.split('\n') if c.strip()]
                    analyze_comments(comments, categorizer)
                else:
                    st.warning("Please enter some comments to analyze.")
        
        elif input_method == "File Upload":
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write("Preview of uploaded data:")
                    st.dataframe(df.head())
                    
                    # Select comment column
                    comment_column = st.selectbox(
                        "Select comment column:", 
                        df.columns, 
                        key="uploaded_comment_column"
                    )

                    
                    if st.button("🔍 Analyze Uploaded Comments"):
                        comments = df[comment_column].dropna().tolist()
                        analyze_comments(comments, categorizer)
                        
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        
        else:  # Sample Data
            st.write("Sample comments from the training dataset:")
            training_data = categorizer.prepare_training_data()
            sample_comments = training_data['comment'].head(10).tolist()
            
            for i, comment in enumerate(sample_comments, 1):
                st.write(f"{i}. {comment}")
            
            if st.button("🔍 Analyze Sample Comments"):
                analyze_comments(sample_comments, categorizer)
    
    with col2:
        st.header("📊 Results")
        
        if st.session_state.analyzed_comments:
            display_results(st.session_state.analyzed_comments, categorizer)
        else:
            st.info("No comments analyzed yet. Please input comments and click 'Analyze Comments'.")

def analyze_comments(comments, categorizer):
    """Analyze comments and store results"""
    with st.spinner("Analyzing comments..."):
        results = []
        
        progress_bar = st.progress(0)
        
        for i, comment in enumerate(comments):
            # Classify comment
            category, confidence = categorizer.classify_comment(comment)
            
            results.append({
                'comment': comment,
                'category': category,
                'confidence': confidence,
                'timestamp': datetime.now()
            })
            
            progress_bar.progress((i + 1) / len(comments))
        
        st.session_state.analyzed_comments = results
        st.success(f"✅ Analyzed {len(comments)} comments!")

def display_results(results, categorizer):
    """Display analysis results"""
    # Create dataframe
    df = pd.DataFrame(results)
    
    # Summary metrics
    st.subheader("📈 Summary Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Comments", len(results))
    
    with col2:
        categories_count = df['category'].nunique()
        st.metric("Categories Found", categories_count)
    
    with col3:
        avg_confidence = df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
    
    with col4:
        high_confidence = len(df[df['confidence'] >= 0.8])
        st.metric("High Confidence", f"{high_confidence}/{len(results)}")
    
    # Category distribution
    st.subheader("📊 Category Distribution")
    
    category_counts = df['category'].value_counts()
    
    # Create pie chart
    fig_pie = px.pie(
        values=category_counts.values,
        names=category_counts.index,
        title="Comment Categories Distribution",
        color_discrete_map={cat: categorizer.categories.get(cat, {}).get('color', '#000000') 
                           for cat in category_counts.index}
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Bar chart
    fig_bar = px.bar(
        x=category_counts.index,
        y=category_counts.values,
        title="Comment Count by Category",
        labels={'x': 'Category', 'y': 'Count'},
        color=category_counts.index,
        color_discrete_map={cat: categorizer.categories.get(cat, {}).get('color', '#000000') 
                           for cat in category_counts.index}
    )

    
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Confidence distribution
    st.subheader("🎯 Confidence Distribution")
    
    fig_conf = px.histogram(
        df,
        x='confidence',
        nbins=20,
        title="Confidence Score Distribution",
        labels={'confidence': 'Confidence Score', 'count': 'Number of Comments'}
    )
    st.plotly_chart(fig_conf, use_container_width=True)
    
    # Detailed results
    st.subheader("📋 Detailed Results")
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        selected_categories = st.multiselect(
            "Filter by categories:",
            options=df['category'].unique(),
            default=df['category'].unique()
        )
    
    with col2:
        min_confidence = st.slider(
            "Minimum confidence:",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1
        )
    
    # Filter data
    filtered_df = df[
        (df['category'].isin(selected_categories)) &
        (df['confidence'] >= min_confidence)
    ]
    
    # Display filtered results
    st.write(f"Showing {len(filtered_df)} out of {len(df)} comments")
    
    for idx, row in filtered_df.iterrows():
        category_info = categorizer.categories.get(row['category'], {})
        category_label = category_info.get('label', row['category'])
        category_color = category_info.get('color', '#000000')
        
        st.markdown(f"""
        <div class="category-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="background-color: {category_color}; color: white; padding: 0.25rem 0.75rem; 
                      border-radius: 15px; font-size: 0.8rem; font-weight: bold;">
                    {category_label}
                </span>
                <span style="color: #6c757d; font-size: 0.9rem;">
                    Confidence: {row['confidence']:.1%}
                </span>
            </div>
            <div class="comment-text">
                "{row['comment']}"
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Export options
    st.subheader("📤 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Download CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name=f"comment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📋 Download JSON"):
            json_data = df.to_json(orient='records', indent=2)
            st.download_button(
                label="Download Results as JSON",
                data=json_data,
                file_name=f"comment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("📈 Generate Report"):
            generate_analysis_report(df, categorizer)

def generate_analysis_report(df, categorizer):
    """Generate a comprehensive analysis report"""
    st.subheader("📈 Analysis Report")
    
    # Calculate statistics
    total_comments = len(df)
    category_counts = df['category'].value_counts()
    avg_confidence = df['confidence'].mean()
    high_confidence_count = len(df[df['confidence'] >= 0.8])
    low_confidence_count = len(df[df['confidence'] < 0.5])
    
    # Generate report text
    report = f"""
    ## Comment Analysis Report
    
    **Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    ### Summary Statistics
    - **Total Comments Analyzed:** {total_comments}
    - **Categories Detected:** {len(category_counts)}
    - **Average Confidence:** {avg_confidence:.2%}
    - **High Confidence Predictions (≥80%):** {high_confidence_count} ({high_confidence_count/total_comments:.1%})
    - **Low Confidence Predictions (<50%):** {low_confidence_count} ({low_confidence_count/total_comments:.1%})
    
    ### Category Breakdown
    """
    
    for category, count in category_counts.items():
        percentage = (count / total_comments) * 100
        category_label = categorizer.categories.get(category, {}).get('label', category)
        report += f"- **{category_label}:** {count} comments ({percentage:.1f}%)\n"
    
    report += f"""
    
    ### Key Insights
    - The most common category is **{categorizer.categories.get(category_counts.index[0], {}).get('label', category_counts.index[0])}** with {category_counts.iloc[0]} comments
    - {high_confidence_count} predictions have high confidence (≥80%)
    - {low_confidence_count} predictions may need manual review (confidence <50%)
    
    ### Recommendations
    """
    
    # Add recommendations based on analysis
    if low_confidence_count > total_comments * 0.2:
        report += "- Consider retraining the model with more diverse training data\n"
    
    if 'Hate/Abuse' in category_counts and category_counts['Hate/Abuse'] > 0:
        report += f"- {category_counts['Hate/Abuse']} comments were classified as hate/abuse and may need immediate attention\n"
    
    if 'Threat' in category_counts and category_counts['Threat'] > 0:
        report += f"- {category_counts['Threat']} comments were classified as Threats and should be reviewed urgently\n"
    
    if 'Irrelevant/Spam' in category_counts and category_counts['Irrelevant/Spam'] > total_comments * 0.3:
        report += "- High Irrelevant/Spam rate detected - consider implementing Irrelevant/Spam filters\n"
    
    st.markdown(report)
    
    # Download report
    st.download_button(
        label="📄 Download Full Report",
        data=report,
        file_name=f"comment_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )

def clear_results():
    """Clear analysis results"""
    st.session_state.analyzed_comments = []
    st.rerun()

# Add this to the sidebar in the main function
def add_clear_button():
    """Add clear results button to sidebar"""
    if st.session_state.analyzed_comments:
        st.divider()
        if st.button("🗑️ Clear Results"):
            clear_results()

# Add this to the main function after the model training section
def classify_comment(self, comment):
    """Classify a single comment"""
    if st.session_state.get("trained_model") is None:
        return self.rule_based_classification(comment)

    try:
        model_type = st.session_state.model_type
        if model_type in ['bert', 'distilbert']:
            pipeline = st.session_state.trained_model
            model = pipeline['model']
            tokenizer = pipeline['tokenizer']
            label_map = pipeline['label_map']
            device = pipeline['device']

            # Tokenize input
            encodings = tokenizer(
                [comment],
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors='pt'
            )
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)

            # Predict
            model.eval()
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
                confidence, pred_idx = probs.max(dim=1)
                prediction = list(label_map.keys())[pred_idx.item()]
                confidence = confidence.item()
            return prediction, confidence
        else:
            processed_comment = self.preprocess_text(comment)
            model = st.session_state.trained_model
            prediction = model.predict([processed_comment])[0]
            confidence = model.predict_proba([processed_comment]).max()
            return prediction, confidence
    except Exception as e:
        st.error(f"Error in classification: {str(e)}")
        return self.rule_based_classification(comment)

if __name__ == "__main__":
    main()