# PDF Chatbot with Flask and GPT-2

This project implements a chatbot that allows users to upload PDF files, extract the text, and interactively ask questions about the content using GPT-2 for language modeling. The project also integrates with AWS S3 for file storage and DynamoDB for user data management.

## Features
- User authentication (Signup/Login/Password Reset)
- Upload PDF files and extract text
- Chatbot interface for asking questions based on the uploaded PDF
- Text embedding using GPT-2
- AWS S3 for storing PDF files
- AWS DynamoDB for storing user responses
- Export recent chats as text files

## Technologies Used
- Flask (Python web framework)
- PyPDF2 (PDF extraction)
- GPT-2 (Natural Language Processing via Hugging Face Transformers)
- AWS S3 (File storage)
- AWS DynamoDB (User data and response storage)
- FAISS (Vector similarity search)
- sklearn (k-Nearest Neighbors)
- boto3 (AWS SDK for Python)

## Requirements
- Python 3.8+
- AWS account with S3 and DynamoDB configured
- API keys for OpenAI (for GPT-2) and AWS

## Python Packages
You can install the required packages using the following command:
```bash
pip install -r requirements.txt
```

## Setup
1. **Clone the repository:**
```bash

git clone https://github.com/oussema68/PDFai.git
cd PDFai
```

2. **Set up environment variables:**

```bash

export OPENAI_API_KEY=your_openai_api_key
export AWS_ACCESS_KEY=your_aws_access_key
export AWS_SECRET_KEY=your_aws_secret_key
export FLASK_SECRET_KEY=your_flask_secret_key
```
Replace your_openai_api_key, your_aws_access_key, your_aws_secret_key, your_flask_secret_key with your actual credentials.


3. **Configure AWS:**

- Set up an S3 bucket for storing PDFs.
- Create a DynamoDB table for user interactions and responses. Update the following variables in your script(bucket_name, region_name, table_name, users_table_name.)


4. **Run the application:**
```bash
flask run
```

## Usage
1. **Sign Up / Login:**
- Access the app at http://127.0.0.1:5000/.
- Sign up with a username and password.
- Log in to access the main functionality.
2. **Upload a PDF:**
- After logging in, upload a PDF file. The file will be uploaded to AWS S3, and the text will be extracted.
3. **Ask Questions:**
- Once the PDF is uploaded. Navigate to the chat interface and ask questions related to the uploaded PDF content.
- The chatbot uses GPT-2 to provide answers based on the PDF.
4. **Recent Chats:**
- View your recent chats from the current session.
- You can also download recent chats as a .txt file.
## Password Reset
- If you forget your password, you can reset it on the /reset_password route.
## Deployment
- The app can be deployed on any cloud service that supports Flask (e.g., AWS, Heroku).
- Ensure environment variables for Flask, OpenAI, and AWS are set correctly before deploying.
## License
**This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.**

## Additional Notes:
Modify the DynamoDB and S3 configurations based on your setup.
