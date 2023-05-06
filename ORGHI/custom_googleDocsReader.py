from llama_index import (
    download_loader,
    GPTSimpleVectorIndex,
)

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
            temperature=0.5,
            model_name="text-davinci-003",
            max_tokens=num_outputs,
        )
    )
    # 2.4: Create an instance of GoogleDocsReader
    reader = GoogleDocsReader()

    # 2.5: Get the list of document ID's from Google 
    googleDocIds = ['12fr-0TQJHw5VRwY0v4Ewv6pAdr-tcw3SSPNnFDwn_5U']

    # 2.6: Load the the googleDocs
    googleDocs = reader.load_data(document_ids=googleDocIds)

    # 2.7: Create an instance of GPTSimpleVectorIndex
    index = GPTSimpleVectorIndex.from_documents(
        googleDocs, service_context=service_context
        )


    index.save_to_disk("GPT_index/googleDoc_index.json")

# Create a function that retireves all available document ids from google drive folder ORGHI Knowledge

def retireve_all_document_ids():
    