# sms_spam_detector
# 📱 SMS Spam Classifier with Gradio Interface

This project demonstrates how to build a machine learning model to classify SMS messages as either **spam** or **ham** (non-spam). Utilizing a pipeline that combines TF-IDF vectorization and Linear Support Vector Classification (LinearSVC), the model is trained on the "SMSSpamCollection" dataset. A user-friendly Gradio interface is provided for real-time predictions.

![SMS Spam Classifier Interface](https://github.com/Govind155/Spam-Classification-using-NLP/blob/main/images/spam_classifier_interface.png)

---

## 🧪 Project Overview

The goal is to develop a model that can accurately classify SMS text messages as either spam or ham. The solution includes:

- **Data Preprocessing**: Cleaning and preparing the SMS dataset.
- **Model Training**: Training a LinearSVC model using TF-IDF vectorization.
- **Gradio Interface**: A web interface allowing users to input an SMS message and get an immediate classification result.

---

## 📦 Dependencies

Ensure you have the following Python packages installed:

```bash
pip install pandas scikit-learn gradio
```
📁 Dataset
The model is trained on the SMSSpamCollection dataset, which contains SMS messages labeled as either "spam" or "ham". If the dataset is not available, a sample dataset is provided in the repository.
🧠 Model Pipeline
The classification pipeline consists of:

TF-IDF Vectorization: Converts text messages into numerical features.

Linear Support Vector Classification (LinearSVC): Classifies the messages based on the TF-IDF features.

🚀 Usage
1. Load the Dataset
python
Copy
Edit
import pandas as pd

# Load the dataset
sms_text_df = pd.read_csv('Resources/SMSSpamCollection.csv', sep='\t', header=None, names=['label', 'text_message'])
2. Train the Model
python
Copy
Edit
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sms_text_df['text_message'], sms_text_df['label'], test_size=0.33, random_state=42)

# Build and train the pipeline
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LinearSVC())
])

text_clf.fit(X_train, y_train)
3. Make Predictions
python
Copy
Edit
def sms_prediction(text):
    prediction = text_clf.predict([text])[0]
    return f'The text message: "{text}", is {"not " if prediction == "ham" else ""}spam.'
4. Launch the Gradio Interface
python
Copy
Edit
import gradio as gr

# Create the Gradio interface
sms_app = gr.Interface(
    fn=sms_prediction,
    inputs=gr.Textbox(lines=2, placeholder="Enter SMS text here...", label="What is the text message you want to test?"),
    outputs=gr.Textbox(label="Our app has determined:")
)

# Launch the app
sms_app.launch(share=True, debug=True)
🔍 Example Usage
Input:
"Congratulations! You've won a $1000 gift card. Claim now!"

Output:
The text message: "Congratulations! You've won a $1000 gift card. Claim now!", is spam.

📂 Repository Structure
bash
Copy
Edit
/sms_spam_detector
│
├── Resources/
│   └── SMSSpamCollection.csv  # Dataset
│
├── app.py                    # Main application script
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
📄 License
This project is licensed under the MIT License.

pgsql
Copy
Edit

Feel free to copy and paste this content into your `README.md` file on GitHub. If you need further customization or additional sections, feel free to ask!
::contentReference[oaicite:0]{index=0}
 



