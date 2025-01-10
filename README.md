# *Fake News Detection using Python and Machine Learning*

## 📌 *Project Overview*
This project focuses on building a *Fake News Detection* system using *Python* and *Machine Learning* techniques. The model classifies news articles as *"Real"* or *"Fake"* based on their content. The project utilizes *Natural Language Processing (NLP)* for text preprocessing and various classification algorithms for model building.

---

## 📂 *Project Structure*

Fake-News-Detection/
│
├── data/                   # Dataset folder
│   └── True.csv            # Real news dataset
│   └── Fake.csv            # Fake news dataset
│
├── notebooks/              # Jupyter Notebooks for development
│   └── fake_news_detection.ipynb
│
├── models/                 # Saved models
│   └── fake_news_model.pkl
│
├── app.py                  # Flask/Streamlit app for deployment
│
└── README.md               # Project documentation


---

## ⚙ *Environment Setup*

1. *Clone the repository:*
   bash
   git clone https://github.com/yourusername/Fake-News-Detection.git
   cd Fake-News-Detection
   

2. *Create a virtual environment and activate it:*
   bash
   python -m venv fake-news-env
   # For Windows
   fake-news-env\Scripts\activate
   # For Mac/Linux
   source fake-news-env/bin/activate
   

3. *Install the required dependencies:*
   bash
   pip install -r requirements.txt
   

4. *Run the Jupyter Notebook:*
   bash
   jupyter notebook
   

---

## 📊 *Dataset*
The dataset used in this project is taken from *Kaggle*. It contains two files:
- *True.csv:* Contains real news articles.
- *Fake.csv:* Contains fake news articles.

You can download the dataset from the following link:
[https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

---

## 🔄 *Data Preprocessing*
The following preprocessing steps were performed on the dataset:
- Converting text to lowercase.
- Removing punctuation and special characters.
- Removing stopwords.
- Tokenization.
- Stemming/Lemmatization.

---

## 🤖 *Machine Learning Model*
The following algorithms were used for building the Fake News Detection model:
1. *Logistic Regression*
2. *Naive Bayes*

The text data was vectorized using *TF-IDF (Term Frequency-Inverse Document Frequency)* to convert text into numerical features.

---

## 📈 *Model Evaluation*
The model performance was evaluated using:
- *Accuracy Score*
- *Confusion Matrix*
- *Precision, Recall, F1 Score*

---

## 🚀 *How to Run the Project*

1. *Run the Jupyter Notebook to train the model:*
   bash
   jupyter notebook notebooks/fake_news_detection.ipynb
   

2. *Run the Flask app for deployment:*
   bash
   python app.py
   

---

## 🛠 *Technologies Used*
- *Python*
- *Pandas*
- *NumPy*
- *Scikit-learn*
- *NLTK*
- *Matplotlib*
- *Seaborn*
- *Flask/Streamlit*

---

## 📚 *Further Improvements*
- Try different machine learning algorithms such as *Random Forest* or *SVM*.
- Perform *hyperparameter tuning* to improve model performance.
- Deploy the project using *Streamlit* or *Heroku* for a user-friendly interface.

---

## 📜 *License*
This project is licensed under the *Apache 2.0 License*. Feel free to use and modify the code.
