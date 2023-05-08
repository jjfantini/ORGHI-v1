import os
import sys

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from IPython.display import Markdown, display
from langchain import OpenAI
from llama_index import (GPTSimpleVectorIndex, LLMPredictor, PromptHelper,
                         ServiceContext, download_loader)

# 1: Create an instance of GoogleDocsReader
googleDocsReader = download_loader("GoogleDocsReader")


# 2: Create a function that generates GPT index from our googleDocs
def generate_googleDoc_index():

    # 2.1: Set prompt_helper parameters
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    # 2.2: define prompt helper
    prompt_helper = PromptHelper(
        max_input_size,
        num_outputs,
        max_chunk_overlap,
        chunk_size_limit=chunk_size_limit,
    )

    # 2.3: define LLM
    llm_predictor = LLMPredictor(
        llm=OpenAI(
            temperature=0.2,
            model_name="text-davinci-003",
            max_tokens=num_outputs,
        )
    )
    # 2.4: Create an instance of GoogleDocsReader
    reader = googleDocsReader()

    # 2.5: Get the list of document ID's from Google 
    googleDocIds = ['12fr-0TQJHw5VRwY0v4Ewv6pAdr-tcw3SSPNnFDwn_5U']

    # 2.6: Load the the googleDocs
    googleDocs = reader.load_data(document_ids=googleDocIds)

    # 2.7: Create an instance of ServiceContext
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    # 2.8: Create an instance of GPTSimpleVectorIndex
    index = GPTSimpleVectorIndex.from_documents(
        googleDocs, service_context=service_context
        )


    index.save_to_disk("GPT_index/googleDoc_index.json")


# TEST - query the chosen document, ask_ai() be moved to a main() file
ask_ai(index_name="GPT_index/googleDoc_index")

# Create a function that retireves all available document ids from google drive folder ORGHI Knowledge

def get_document_ids(folder_name):

    try:
        # Set up the credentials
        SCOPES = ['https://www.googleapis.com/auth/drive']
        SERVICE_ACCOUNT_FILE = 'credentials.json'

        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES
            )

        # Build the Google Drive API client
        service = build('drive', 'v3', credentials=creds)

        # Search for the folder by name
        query = "mimeType='application/vnd.google-apps.folder' and trashed = false and name='" + folder_name + "'"
        results = service.files().list(q=query, fields="nextPageToken, files(id, name)").execute()
        items = results.get('files', [])

        if not items:
            print(f'No folder found with the name "{folder_name}".')
            return None
        else:
            folder_id = items[0]['id']

        # Get all documents inside the folder
        query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.document' and trashed = false"
        results = service.files().list(q=query, fields="nextPageToken, files(id, name)").execute()
        items = results.get('files', [])

        if not items:
            print(f'No documents found in the folder "{folder_name}".')
            return None
        else:
            document_ids = [item['id'] for item in items]
            return document_ids

    except HttpError as error:
        print(f'An error occurred: {error}')
        return None

# Example usage
folder_name = 'ORGHI Knowledge'
document_ids = get_document_ids(folder_name)
print(f'Document IDs in the folder "{folder_name}": {document_ids}')



# TESTING:
from __future__ import print_function

import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly']


def get_docs_from_folder(folder_id, no_docs):
    
      """
    Retrieves a specified number of documents from a Google Drive folder.

    This function demonstrates basic usage of the Drive v3 API. It prints the names and ids of the first 'no_docs' files
    the user has access to within the specified folder.

    Args:
        folder_id (str): The ID of the Google Drive folder to retrieve documents from.
        no_docs (int): The number of documents to retrieve from the folder.

    Returns:
        None

    Raises:
        HttpError: If an error occurs while making the API request.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('drive', 'v3', credentials=creds)

        # Call the Drive v3 API
        results = service.files().list(
            q=f"'{folder_id}' in parents",
            pageSize=no_docs, fields="nextPageToken, files(id, name)").execute()
        items = results.get('files', [])

        if not items:
            print('No files found.')
            return
        print('Files:')
        for item in items:
            print(u'{0} ({1})'.format(item['name'], item['id']))
    except HttpError as error:
        # TODO(developer) - Handle errors from drive API.
        print(f'An error occurred: {error}')


if __name__ == '__main__':
    main()