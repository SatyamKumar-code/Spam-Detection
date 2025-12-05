# Email Spam Detection

A web application for detecting whether an email message is spam or safe using a machine learning model.

## Features

- Classifies email messages as "spam" or "safe"
- Interactive web interface with modern UI
- Displays a table of all emails and their classifications

## Project Structure

- `templates/index.html` - Main web interface
- `data/spam.csv` - Sample dataset of emails
- `app.py` - Flask backend

## Live Demo

- **Frontend:** [https://ems-bice-rho.vercel.app/](https://ems-bice-rho.vercel.app/)
- **Backend API:** [https://esd-z761.onrender.com/](https://esd-z761.onrender.com/)

## Setup (Local Development)

1. **Clone the repository**  
   ```
   git clone https://github.com/Saurabhtbj1201/email-spam-detection
   cd email_spam_detection
   ```

2. **Install dependencies**  
   Make sure you have Python 3.x and pip installed.  
   ```
   pip install -r requirements.txt
   ```

3. **Run the Flask app**  
   ```
   python app.py
   ```

4. **Open the web interface**  
   Open `templates/index.html` in your browser, or deploy the frontend as shown below.

## Deployment

- **Frontend:** Deployed on [Vercel](https://vercel.com/)  
  (see `templates/index.html`)

- **Backend:**
  Deployed on [Render](https://render.com/)  
  [https://esd-z761.onrender.com/](https://esd-z761.onrender.com/)

## Usage

- Enter an email message in the text area and click "Check" to classify it.
- The result will display whether the email is spam or safe.
- All emails and their classifications are shown in a table (if implemented in backend).

## API Endpoints

- `POST /predict`  
  Request JSON: `{ "message": "your email text" }`  
  Response JSON: `{ "message": "...", "prediction": "spam" | "safe" }`

- `GET /`  
  Returns a simple status message.

- `GET /emails`  
  (If implemented) Returns all emails and their classifications.
