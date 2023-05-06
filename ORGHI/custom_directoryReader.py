import os
import sys

from dotenv import load_dotenv
from IPython.display import Markdown, display
from langchain import OpenAI
from llama_index import (
    GPTListIndex,
    GPTSimpleVectorIndex,
    LLMPredictor,
    PromptHelper,
    ServiceContext,
    SimpleDirectoryReader,
    readers,
)


# 1: SET API KEY
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# 2: Create a function to generate GPT Index for our data
def generate_dir_index(directory_path):
    # set directory path - acces knowledge folder within the ORGHI/ folder
    abs_directory_path = os.path.abspath(directory_path)

    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    # define prompt helper
    prompt_helper = PromptHelper(
        max_input_size,
        num_outputs,
        max_chunk_overlap,
        chunk_size_limit=chunk_size_limit,
    )

    # define LLM
    llm_predictor = LLMPredictor(
        llm=OpenAI(
            temperature=0.5,
            model_name="text-davinci-003",
            max_tokens=num_outputs,
        )
    )

    # load data into documents object
    documents = SimpleDirectoryReader(
        abs_directory_path,
        recursive=True,
    ).load_data()

    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )
    index = GPTSimpleVectorIndex.from_documents(
        documents, service_context=service_context
    )

    index.save_to_disk("GPT_index/directory_index.json")

    return index


# 3: Create Function to interact with the files
def ask_ai(index_name = "index"):
    # Create an index file name by appending ".json" to the given index_name
    index = f"{index_name}.json"

    # Load the GPTSimpleVectorIndex object from the specified index file on disk
    index = GPTSimpleVectorIndex.load_from_disk(index)

    while True:
        query = input("What do you want to ask? (Type 'exit' to quit) ")

        # Check if the user wants to exit
        if query.lower() == "exit":
            break

        response = index.query(query)
        display(Markdown(f"Response: <b>{response.response}</b>"))


# 4: Start Conversation
# 4.1: Create the knowledge index
generate_dir_index("knowledge/")

# 4.2: Ask LLM about the knowledge
ask_ai()
