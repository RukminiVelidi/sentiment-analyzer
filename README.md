End-to-End Sentiment Analysis Web Application
A full-stack web application that performs real-time sentiment analysis on user-provided text. The project features a modern frontend built with HTML/Tailwind CSS and a powerful Python backend powered by a custom-trained Scikit-learn machine learning model served via a Flask API.

🚀 Live Demo
You can view the live application here: https://RukminiVelidi.github.io/sentiment-analyzer/

(Note: The backend is not yet deployed, so the live demo will show a "Connection Error." This is expected until the Python server is hosted on a service like Render.)

(To add a screenshot: Take a screenshot of your running application, add the image file to your project folder, commit, and push it. Then, replace the placeholder link above with the direct link to the image on GitHub.)

Features
Real-Time Predictions: Instantly analyze text and receive sentiment predictions (Positive, Negative, or Neutral).

Confidence Score: Each prediction is accompanied by a confidence score, indicating the model's certainty.

Intelligent Neutral Handling: The application intelligently interprets low-confidence (below 85%) predictions as "Neutral," providing a more robust user experience for nuanced text.

Interactive UI: A sleek, modern, and responsive user interface with an animated background and loading states.

Full-Stack Architecture: Demonstrates a complete end-to-end workflow, from model training to API deployment and frontend integration.

Tech Stack
Frontend: HTML5, Tailwind CSS, JavaScript (ES6+)

Backend: Python, Flask, Flask-CORS

Machine Learning: Scikit-learn, Pandas, Joblib

Environment: Python Virtual Environment (venv)

Local Setup & Installation
Follow these instructions to set up and run the project on your local machine.

Prerequisites
Python 3.8 or newer

pip (Python package installer)

1. Clone the Repository
First, clone this repository to your local machine.

git clone [https://github.com/RukminiVelidi/sentiment-analyzer.git](https://github.com/RukminiVelidi/sentiment-analyzer.git)
cd sentiment-analyzer

2. Set Up the Backend
The backend server and the model training process are managed in the backend directory.

# Navigate to the backend directory
cd backend

# Create a Python virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

3. Install Dependencies
Install all the necessary Python libraries using the requirements.txt file.

pip install -r requirements.txt

4. Train the Model
Run the training script to download the dataset and train the sentiment_pipeline.pkl model file. This step will take a few minutes.

python train_model.py

5. Run the Backend Server
Once the model is trained, start the Flask API server.

python app.py

The server will start running on http://localhost:5001.

6. Launch the Frontend
With the backend server running, open the index.html file in your web browser. The application is now fully functional locally.

Project Structure
sentiment-analyzer/
│
├── .gitignore          # Specifies files for Git to ignore
├── README.md           # This project documentation
├── index.html          # The complete frontend application
│
└── backend/
    ├── app.py          # Flask API server
    ├── train_model.py  # Script to train the ML model
    └── requirements.txt# List of Python dependencies
