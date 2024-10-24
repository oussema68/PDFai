from flask import Flask, render_template, request, redirect, url_for, session, Response
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model
import torch
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
import numpy
from sklearn.neighbors import NearestNeighbors

# Uncomment these imports if they are used later in your code
# from langchain.chains.question_answering import load_qa_chain
# from langchain_community.callbacks import get_openai_callback

import io
import boto3
import botocore
import uuid
from werkzeug.security import check_password_hash, generate_password_hash

# Load API keys from environment variables for security
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# AWS credentials should also be loaded from environment variables
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
region_name = os.getenv('REGION_NAME')   # Define your AWS region
bucket_name = os.getenv('BUCKET_NAME')  # Define your S3 bucket name

# Initialize DynamoDB and S3 resources using the provided AWS credentials
dynamodb = boto3.resource(
    'dynamodb',
    region_name=region_name,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

s3 = boto3.client(
    's3',
    region_name=region_name,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# Define the DynamoDB table for storing user responses
table_name = 'UserResponses'  # Replace with your actual DynamoDB table name
dynamodb_table = dynamodb.Table(table_name)

# Define the DynamoDB table for user accounts
users_table_name = 'UserAccounts'  # Replace with your actual DynamoDB table name
users_table = dynamodb.Table(users_table_name)

# Create a Flask app and set the secret key for session management
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY')

def get_user_id_by_username(username):
    """
    Retrieve the user ID based on the provided username from the DynamoDB users table.

    Args:
        username (str): The username for which to retrieve the user ID.

    Returns:
        str or None: The user ID if found, otherwise None.
    """
    try:
        # Attempt to get the item from the users table using the provided username
        response = users_table.get_item(Key={'username': username})
        return response.get('Item', {}).get('user_id')  # Return the user ID if found
    except users_table.meta.client.exceptions.ClientError as e:
        print(f"DynamoDB Error: {e}")  # Log any DynamoDB errors
        return None


def generate_user_id():
    """
    Generate a new user ID or retrieve the existing one from the session.

    Returns:
        str: The user ID (either new or existing).
    """
    # Check if user already has a session ID
    if 'user_id' in session:
        return session['user_id']

    # Generate a new UUID as the user ID
    new_user_id = str(uuid.uuid4())
    session['user_id'] = new_user_id  # Store the new user ID in the session
    return new_user_id

def get_recent_chats(user_id):
    """
    Retrieve the most recent chats for a specific user.

    Args:
        user_id (str): The ID of the user for whom to retrieve chats.

    Returns:
        list: A list of the most recent chat responses, limited to the last 20.
    """
    responses = get_user_responses(user_id)
    return responses[-20:]  # Adjust the number as per your preference

def get_username_by_user_id(user_id):
    """
    Retrieve the username associated with a given user ID.

    Args:
        user_id (str): The user ID to look up.

    Returns:
        str or None: The username if found, otherwise None.
    """
    try:
        response = users_table.get_item(Key={'user_id': user_id})
        return response.get('Item', {}).get('username')  # Return the username if found
    except users_table.meta.client.exceptions.ClientError as e:
        print(f"DynamoDB Error: {e}")  # Log any DynamoDB errors
        return None

def get_user_responses(user_id):
    """
    Retrieve the responses associated with a specific user.

    Args:
        user_id (str): The user ID for which to retrieve responses.

    Returns:
        list: A list of responses if found, otherwise an empty list.
    """
    if not user_id:
        # Handle the case where 'user_id' is empty or missing
        return []

    try:
        # Retrieve user-specific responses from DynamoDB
        response = dynamodb_table.get_item(Key={'user_id': user_id})
        return response.get('Item', {}).get('responses', [])  # Return the list of responses
    except dynamodb.meta.client.exceptions.ClientError as e:
        print(f"DynamoDB Error: {e}")  # Log any DynamoDB errors
        return []  # Return an empty list if there's an error


def set_user_responses(user_id, responses):
    """
    Set user-specific responses in DynamoDB, limiting the number of responses stored.

    Args:
        user_id (str): The ID of the user.
        responses (list): The list of responses to store.
    """
    # Limit the number of responses to store for each user
    max_responses = 20  # Adjust the number as per your preference
    responses = responses[-max_responses:]  # Keep only the most recent responses

    # Set user-specific responses in DynamoDB
    dynamodb_table.put_item(Item={'user_id': user_id, 'responses': responses})

def generate_txt_data(responses):
    """
    Generate plain text data from the list of responses.

    Args:
        responses (list): The list of response dictionaries.

    Returns:
        str: Formatted plain text containing recent chats.
    """
    # Generate plain text data from the list of responses
    txt_data = "Recent Chats:\n\n"
    for response in responses:
        txt_data += f'Question: {response["question"]}\nAnswer: {response["answer"]}\n\n'
    return txt_data

def upload_file_to_s3(file, user_id):
    """
    Upload a file to S3 with a key based on the user ID.

    Args:
        file: The file object to upload.
        user_id (str): The ID of the user uploading the file.

    Returns:
        str: The key of the uploaded file in S3.
    """
    file_key = f"{user_id}_{file.filename}"  # Create a unique file key using user ID
    s3.upload_fileobj(file, bucket_name, file_key)  # Upload the file to S3
    return file_key

def download_file_from_s3(file_key):
    """
    Download a file from S3 based on the provided file key.

    Args:
        file_key (str): The key of the file to download.

    Returns:
        bytes or None: The contents of the file if found, otherwise None.
    """
    if file_key:
        file_stream = io.BytesIO()  # Create an in-memory byte stream
        try:
            s3.download_fileobj(bucket_name, file_key, file_stream)  # Download the file
            return file_stream.getvalue()  # Return the contents of the file
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                # Object not found, handle it gracefully
                print(f"S3 Object with key '{file_key}' not found.")
            else:
                # Handle other S3 client errors
                print(f"S3 Error: {e}")
    else:
        # Handle the case where file_key is None
        print("file_key is None. Cannot download file.")
    return None


def login_user(username, password):
    """
    Authenticate a user by username and password.

    Args:
        username (str): The username of the user.
        password (str): The password provided by the user.

    Returns:
        bool: True if login is successful, False otherwise.
    """
    if username and password:
        try:
            # Retrieve user information from DynamoDB
            response = users_table.get_item(Key={'username': username})
            user_data = response.get('Item')

            if user_data and check_password_hash(user_data['password'], password):
                # Store user_id in session upon successful login
                session['user_id'] = user_data['user_id']
                return True  # User exists and password is correct
        except users_table.meta.client.exceptions.ClientError as e:
            print(f"DynamoDB Error: {e}")

    return False  # User does not exist or password is incorrect

def signup_user(username, password):
    """
    Register a new user with a username and password.

    Args:
        username (str): The desired username for the new user.
        password (str): The desired password for the new user.

    Returns:
        str or None: The user_id if signup is successful, None otherwise.
    """
    if username and password:
        # Validate password strength
        if (
            len(password) >= 8
            and any(char.isdigit() for char in password)
            and any(char.isupper() for char in password)
            and any(char.islower() for char in password)
            and any(char in "!@#$%^&*()-_=+[]{}|;:'\",.<>/?`~" for char in password)
        ):
            try:
                # Check if the username already exists in DynamoDB
                response = users_table.get_item(Key={'username': username})
                existing_user = response.get('Item')

                if existing_user:
                    return None  # Username already exists

                # Create a new user in DynamoDB
                hashed_password = generate_password_hash(password)
                new_user_id = str(uuid.uuid4())
                users_table.put_item(Item={'user_id': new_user_id, 'username': username, 'password': hashed_password})

                # Store user_id in session upon successful signup
                session['user_id'] = new_user_id

                return new_user_id
            except users_table.meta.client.exceptions.ClientError as e:
                print(f"DynamoDB Error: {e}")
                return None  # An error occurred during signup
        else:
            return None  # Password does not meet the minimum requirements

    return None

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    """
    Handle password reset requests for users.

    Returns:
        str: The rendered HTML for the reset password page.
    """
    error_message = None
    success_message = None

    if request.method == 'POST':
        username = request.form.get('resetUsername')
        new_password = request.form.get('newPassword')
        confirm_password = request.form.get('confirmPassword')

        # Validate new password requirements
        if (
            len(new_password) >= 8
            and any(char.isdigit() for char in new_password)
            and any(char.isupper() for char in new_password)
            and any(char.islower() for char in new_password)
            and any(char in "!@#$%^&*()-_=+[]{}|;:'\",.<>/?`~" for char in new_password)
        ):
            if new_password != confirm_password:
                error_message = "Passwords don't match"
            else:
                hashed_password = generate_password_hash(new_password)

                try:
                    # Check if the username exists in the table
                    response = users_table.get_item(Key={'username': username})

                    if 'Item' in response:
                        success_message = f"Password changed successfully for {username}."

                        # Update the password for the provided username
                        users_table.put_item(Item={'user_id': response['Item']['user_id'], 'username': username, 'password': hashed_password})
                    else:
                        # Username not found, display an error message
                        error_message = "Username does not exist."

                except users_table.meta.client.exceptions.ClientError as e:
                    print(f"DynamoDB Error: {e}")
                    error_message = "An error occurred. Please try again."
        else:
            error_message = "Password does not meet the minimum requirements."

    return render_template('reset.html', error_message=error_message, success_message=success_message)


@app.route('/recent_chats', methods=['GET'])
def recent_chats():
    """
    Retrieve and display the recent chats for the logged-in user.

    Returns:
        str: The rendered HTML for the recent chats page.
    """
    # Retrieve the current user ID using the session cookie
    user_id = session.get('user_id', None)
    if user_id is None:
        return redirect(url_for('login_create_account'))

    # Retrieve recent chats for the current user
    recent_chats = get_recent_chats(user_id)

    # Render the recent_chats.html template with the recent chat data
    return render_template('recent_chats.html', recent_chats=recent_chats)

@app.route('/download_recent_chats', methods=['GET'])
def download_recent_chats():
    """
    Allow the user to download their recent chats as a text file.

    Returns:
        Response: A downloadable text file containing recent chats.
    """
    # Retrieve the current user ID using the session cookie
    user_id = session.get('user_id', None)
    if user_id is None:
        return redirect(url_for('login_create_account'))

    # Retrieve recent chats for the current user
    recent_chats = get_recent_chats(user_id)

    # Generate plain text data from recent chats
    txt_data = generate_txt_data(recent_chats)

    # Return the text data as a downloadable file
    response = Response(txt_data, content_type='text/plain')
    response.headers["Content-Disposition"] = f"attachment; filename=recent_chats_{user_id}.txt"
    return response

@app.route('/index', methods=['GET', 'POST'])
def index():
    """
    Handle the file upload process and redirect to the questioning phase.

    Returns:
        str: The rendered HTML for the index page.
    """
    if request.method == 'POST':
        pdf_file = request.files['pdf_file']
        if pdf_file:
            # Use session cookie to identify the user
            user_id = session.get('user_id', generate_user_id())
            session['user_id'] = user_id  # Ensure user_id is set in session

            # Upload PDF file to S3
            pdf_key = upload_file_to_s3(pdf_file, user_id)

            # Store user-specific data in session
            session['pdf_key'] = pdf_key

            # Set initial user-specific responses in session
            set_user_responses(user_id, [])

            return redirect(url_for('continuous_questioning'))

    return render_template('index.html')

    
@app.route('/continuous_questioning', methods=['GET', 'POST'])
def continuous_questioning():
    # Retrieve the current user ID using session cookie
    user_id = session.get('user_id', None)
    if user_id is None:
        return redirect(url_for('index'))

    # Retrieve user-specific data from session
    pdf_key = session.get('pdf_key', None)

    # Retrieve user-specific responses from DynamoDB
    responses = get_user_responses(user_id)

    # Initialize a flag to indicate if the answer is processing
    answer_processing = False

    if request.method == 'POST':
        query = request.form['query']

        if query and pdf_key:
            # Set the flag to indicate that the answer is processing
            answer_processing = True

            # Download PDF file from S3
            pdf_content = download_file_from_s3(pdf_key)

            if pdf_content:
                # Read the PDF file
                with io.BytesIO(pdf_content) as pdf_file:
                    pdf_reader = PdfReader(pdf_file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()

                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len
                    )
                    chunks = text_splitter.split_text(text=text)

                    # Replace VectorStore with the appropriate vector store or embedding mechanism
                    # based on your specific use case
                    # Load pre-trained GPT-2 tokenizer and model
                    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                    model = GPT2Model.from_pretrained("gpt2")

                    # Set the tokenizer's pad token to the end-of-sequence token
                    tokenizer.pad_token = tokenizer.eos_token

                   

                    def get_gpt2_embeddings(text):
                        # Tokenize input text
                        tokenized_text = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                        input_ids = tokenized_text["input_ids"]
                        position_ids = tokenized_text["attention_mask"].cumsum(dim=1) - 1

                        # Obtain embeddings from GPT-2 model
                        with torch.no_grad():
                            embeddings = model(input_ids, position_ids=position_ids).last_hidden_state.mean(dim=1).squeeze()


                        return embeddings.numpy()

                    # Example usage
                    
                    embeddings = get_gpt2_embeddings(text)
                    # Create a custom embedding object
                    class CustomEmbedding:
                        def embed_documents(self, texts):
                            # Implement the logic to embed the documents
                            # You may need to use a different method based on your specific use case
                            pass

                    # Create an instance of the custom embedding
                    custom_embedding = CustomEmbedding()

                    class VectorStore:
                        def __init__(self, vectors):
                            self.vectors = vectors
                            self.knn_model = NearestNeighbors(n_neighbors=3, algorithm='auto', metric='cosine')
                            self.knn_model.fit(self.vectors)

                        def similarity_search(self, query, k=3):
                            query_embedding = get_gpt2_embeddings(query)
                            _, indices = self.knn_model.kneighbors([query_embedding], n_neighbors=k)
                            return indices[0]
                        

                    VectorStore = VectorStore(custom_embedding)

                    docs = VectorStore.similarity_search(query=query, k=3)

                    llm = GPT2LMHeadModel

                    chain = LLMChain(prompt=prompt, llm=llm) 
                    # Replace with the appropriate GPT-2 chain type
                    #with get_openai_callback() as cb:
                    #    response = chain.run(input_documents=docs, question=query)

                    # Inside your route or function
                    response = chain.run(input_documents=docs, question=query)

                    # Append the question and its answer as a pair to user-specific responses
                    responses.append({'question': query, 'answer': response})

                    # Update user-specific responses in DynamoDB
                    set_user_responses(user_id, responses)

            return render_template('continuous_questioning.html', responses=responses, answer_processing=answer_processing)

    return render_template('continuous_questioning.html', responses=responses, answer_processing=answer_processing)


@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if request.method == 'POST':
        pdf_file = request.files['pdf_file']
        if pdf_file:
            # Use session cookie to identify the user
            user_id = session.get('user_id', generate_user_id())
            session['user_id'] = user_id  # Ensure user_id is set in session

            # Upload PDF file to S3
            pdf_key = upload_file_to_s3(pdf_file, user_id)

            # Store S3 key in session
            session['pdf_key'] = pdf_key

            # Clear user-specific responses in session
            set_user_responses(user_id, [])

    # Redirect to continuous_questioning route after uploading
    return redirect(url_for('continuous_questioning'))


@app.route('/', methods=['GET', 'POST'])
def login_create_account():
    if request.method == 'POST':
        login_username = request.form.get('loginUsername')
        login_password = request.form.get('loginPassword')
        signup_username = request.form.get('signupUsername')
        signup_password = request.form.get('signupPassword')
        confirm_password = request.form.get('confirmPassword')

        if login_username and login_password:
            # Attempt to log in the user
            if login_user(login_username, login_password):
                # Successful login, set user_id in session
                session['user_id'] = get_user_id_by_username(login_username)
                print("Successful login. Redirecting to index.")
                return redirect(url_for('index'))
            else:
                # Invalid login, show error message
                login_error = 'Invalid username or password'
                print("Invalid login")
        elif signup_username and signup_password and signup_password == confirm_password:
            # Check if the user already exists
            if get_user_id_by_username(signup_username):
                signup_error = 'Username already exists'
            else:
                # User doesn't exist, proceed with signup
                user_id = signup_user(signup_username, signup_password)
                if user_id:
                    # Successful signup, set user_id in session
                    session['user_id'] = user_id
                    print("Successful signup. Redirecting to index.")
                    return redirect(url_for('index'))
                else:
                    # An error occurred during signup
                    signup_error = 'Password does not meet the minimum requirements'
        elif signup_password != confirm_password:
            signup_error = "Passwords don't match"
        else:
            # Show error message for incomplete form
            form_error = 'Please fill out the form completely'

    # Explicitly return the rendered template for the case where the form is not processed
    return render_template('login_create_account.html', login_error=locals().get('login_error', None),
                           signup_error=locals().get('signup_error', None),
                           form_error=locals().get('form_error', None))


@app.route('/logout', methods=['GET'])
def logout():
    # Clear the user_id and pdf_key from the session to log out the user
    session.pop('user_id', None)
    session.pop('pdf_key', None)
    return redirect(url_for('login_create_account'))


if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Change the port number as needed
