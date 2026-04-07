# 🎫IT Support Ticket Classification & Prioritization:
A complete end-to-end machine learning solution for automatically classifying IT service tickets into 8 categories and predicting priority levels for customer support requests using Python, NLP, and Machine Learning.

🔍 Project Summary:
This project transforms raw support tickets into actionable insights. It is designed for IT teams and interns to quickly understand ticket types and priorities, enabling faster resolutions.

Key Features:
🏷️ IT Ticket Classification: Hardware, HR Support, Access, Storage, Miscellaneous, Purchase, Internal Project, Administrative Rights
🚨 Priority Prediction: High / Medium / Low for customer support tickets
🧹 Text Preprocessing Pipeline using NLTK: tokenize → stopword removal → lemmatization → TF-IDF
🤖 Model Benchmarking: Logistic Regression, Linear SVM, Random Forest, Naive Bayes
📊 Visual Dashboard: KPI cards, confusion matrices, model comparison charts
💾 Export models as pickles and generate a ZIP of all project outputs

Dataset Details:
| Dataset                                 | Rows   | Columns | Target                                |
| --------------------------------------- | ------ | ------- | ------------------------------------- |
| `all_tickets_processed_improved_v3.csv` | 47,837 | 2       | IT Ticket Category (8 classes)        |

IT Categories: Hardware · HR Support · Access · Miscellaneous · Storage · Purchase · Internal Project · Administrative Rights
Priority Levels: 🔴 High · 🟡 Medium · 🟢 Low

⚡ Tech Stack:
| Layer             | Library / Tool / Description                                                   |
| ----------------- | ------------------------------------------------------------------------------ |
| **Language**      | Python 3.10                                                                    |
| **NLP**           | nltk — tokenization, stopwords removal, lemmatization, optional stemming       |
| **Features**      | TF-IDF Vectorizer — 15k features, ngram range (1–3), sublinear TF              |
| **Models**        | LogisticRegression, LinearSVC, RandomForestClassifier, MultinomialNB           |
| **Evaluation**    | classification_report, confusion_matrix, score metrics, heatmap visualizations |
| **Visualization** | matplotlib, seaborn, wordcloud                                                 |
| **Environment**   | Jupyter Notebook / Google Colab                                                |
| **Export**        | pickle, zipfile                                                                |

🧹 Text Preprocessing Pipeline:
| Step | Stage            | Description                                                             |
| ---- | ---------------- | ----------------------------------------------------------------------- |
| 1    | Clean Text       | Remove URLs, emails, numbers, and placeholders from raw ticket text     |
| 2    | Tokenize         | Split sentences into individual words using NLTK                        |
| 3    | Filter Stopwords | Remove common NLTK stopwords + domain-specific irrelevant words         |
| 4    | Lemmatize / Stem | Normalize words to their base/root form for better feature consistency  |
| 5    | Vectorize        | Convert text into numerical features using TF-IDF                       |
| 6    | Predict          | Classify tickets and generate confidence scores using trained ML models |

📈 Model Performance:
| Model                 | Accuracy | Precision | Recall | Weighted F1 |
| --------------------- | -------- | --------- | ------ | ----------- |
| Logistic Regression   | 0.8506   | 0.8552    | 0.8506 | 0.8504      |
| Linear SVM            | 0.8485   | 0.8498    | 0.8485 | 0.8484      |
| Random Forest         | 0.8374   | 0.8422    | 0.8374 | 0.8374      |
| Naive Bayes           | 0.7720   | 0.8031    | 0.7720 | 0.7682      |


support-ticket-classifier/
│
├── 📓 support_ticket_classifier.ipynb   ← Full notebook (13 steps)
│
├── 📊 outputs/
│   ├── IT_dashboard.png                 
│   ├── Wordclouds.png                  
│   ├── confusion_matrix_it.png        
│   ├── Model_comparison(1).png          
│   ├── IT Ticket Vloume by category.png        
│   ├── IT Ticket length distribution.png       
│   └── Ticket Category distribution.png             
│
├── 📄 README.md
├── 📄 requirements.txt
└── 📄 .gitignore

model_metrics.csv              ← All model scores in one CSV
ticket_classifier_outputs.zip  ← Everything above bundled for download

📓 Notebook Walkthrough — 13 Steps
| Step | Title                        | What It Does                                                                          |
| ---- | ---------------------------- | ------------------------------------------------------------------------------------- |
| 1    | Install & Import Libraries   | `pip install nltk wordcloud`; all sklearn/matplotlib imports                          |
| 2    | Load Datasets                | Upload CSV; auto-detects IT dataset by column names                                   |
| 3    | Exploratory Data Analysis    | Category bar chart, ticket-type pie, priority bars, word-length histogram             |
| 4    | Data Quality Audit           | Cross-tab analysis                                                                    |
| 5    | Text Preprocessing with NLTK | `clean_text()` → regex cleaning → `word_tokenize()` → stopword filter → lemmatize     |
| 6    | Word Clouds                  | Per-category word clouds showing most discriminative terms                            |
| 7    | Model Training — IT Category | TF-IDF + 4 models; stratified 80/20 split; full accuracy/F1 score table               |
| 8    | Evaluation                   | `classification_report` for both tasks; styled confusion matrices with % annotations  |
| 9    | Cross-Validation             | 5-fold `StratifiedKFold` on a pipeline; fill-between confidence bands plotted         |
| 10   | Feature Importance           | Logistic Regression coefficients → top 10 TF-IDF terms per category (2×4 grid)        |
| 11   | Live Inference               | classifies IT Ticket with category and prediction probability scores                  |
| 12   | Executive Dashboard          | 4 KPI cards + category bars + CV plot + per-class F1 heatmap in one figure            |
| 13   | Save & Download              | Pickle all models + vectorizers + label encoders → ZIP download                       |

📈 Output Charts:
| File                       | Description                                                                                         |
| -------------------------- | --------------------------------------------------------------------------------------------------- |
| Length distribution.png    | IT ticket-length histogram                                               |
| Wordcloud.png              | Top keywords for each of the 8 IT categories                                                        |
| confusion_matrix_it.png    | Side-by-side heatmaps — IT (8×8) with counts and % annotations                                      |
| Model_comparison(1).png    | Grouped bar chart: Accuracy + Weighted F1 for all 4 models across IT category tasks                 |
| IT_dashboard.png           | Full KPI dashboard: 4 cards + category bars + CV plot + per-class F1 heatmap in one figure          |
| IT Volume by category.png  | Bar chart showing the total number of IT tickets per category, highlighting category-wise volume    |
| Category Distribution.png  | Bar chart showing the distribution of IT tickets across all 8 categories                            |

🔍 Live Inference:
->Classify an IT service ticket → category + confidence
->Single IT Ticket Prediction
.Enter IT ticket description:requesting for meeting requesting meeting hi please help follow equipments cable pc cord plug
.Predicted IT Category: Hardware

#Returns:
#Prediction Probabilities
#{
#"Access":"0.61%"
#"Administrative rights":"0.92%"
#"HR Support":"1.09%"
#"Hardware":"95.27%"
#"Internal Project":"0.28%"
#"Miscellaneous":"0.60%"
#"Purchase":"0.71%"
#"Storage":"0.52%"
#}

🚀 Quick Start:
#1. Clone the repo
git clone https://github.com/YOUR_USERNAME/support-ticket-classifier.git
cd support-ticket-classifier

#2. Install dependencies
pip install -r requirements.txt

#3. Launch notebook
jupyter notebook support_ticket_classifier.ipynb

📄 License
This project was completed as part of the Machine Learning Internship Program at Future Interns, focusing on real-world, industry-relevant NLP applications.

Made by Jeevitha | Future Interns Machine Learning Track
