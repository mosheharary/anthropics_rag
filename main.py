import os
from jinja2 import Environment, FileSystemLoader
from tqdm.auto import tqdm
import json
import tiktoken
from openai import OpenAI
from langchain_openai import ChatOpenAI
from openai import OpenAI
import os
import PyPDF2
from pinecone import Pinecone
from typing import Optional, List
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
from contextlib import redirect_stderr

#from pinecone_utility import PineconeUtility

CHUNK_SIZE = 7000
CHUNK_OVERLAP = 500
TEXT_EMBEDDING_MODEL = "text-embedding-3-large"
GPT_MODEL = "gpt-4o-mini"

INPUT_TOKEN_PRICE = 0.15/1e6
OUTPUT_TOKEN_PRICE = 0.6/1e6

PDF_FILEPATHS = r"./docs"
TKT_FILEPATHS = r"./txt"
CHUNKS_SAVE_PATH = r"./chunks"
CHUNKS_EMBEDDINGS_SAVE_PATH=r"./chunks_embeddings"
DOCUMENT_TYPE = "skybox"


class RagAgent:
    def __init__(self, index_name, pinecone, openai_client):
  # load pinecone index
        self.index = pinecone.Index(index_name)
        self.openai_client = openai_client

    def query_pinecone(self, query, top_k=2, include_metadata: bool = True):
        query_embedding = get_embedding(query, TEXT_EMBEDDING_MODEL, self.openai_client)
        query_response = self._query_pinecone_index(query_embedding, top_k=top_k, include_metadata=include_metadata)
        return self._extract_info(query_response)


    def _query_pinecone_index(self, 
        query_embedding: list, top_k: int = 2, include_metadata: bool = True
        ) -> dict[str, any]:
        query_response = self.index.query(
            vector=query_embedding, top_k=top_k, include_metadata=include_metadata, 
        )
        return query_response
 
    def _extract_info(self, response) -> Optional[dict]:
        """extract data from pinecone query response. Returns dict with id, text and chunk idx"""
        if response is None: return None
        res_list = []
        for resp in response["matches"]:
            _id = resp["id"]
            res_list.append(
            {
                "id": _id,
                "chunk_text": resp["metadata"]["chunk_text"],
                "chunk_idx": resp["metadata"]["chunk_idx"],
            })
   
        return res_list
 
    def _combine_chunks(self, chunks_bm25, chunks_vector_db, top_k=20):
        """given output from bm25 and vector database, combine them to only include unique chunks"""
        retrieved_chunks = []
        for chunk1, chunk2 in zip(chunks_bm25, chunks_vector_db):
            if chunk1 not in retrieved_chunks:
                retrieved_chunks.append(chunk1)
                if len(retrieved_chunks) >= top_k:
                    break
            if chunk2 not in retrieved_chunks:
                retrieved_chunks.append(chunk2)
                if len(retrieved_chunks) >= top_k:
                    break
        return retrieved_chunks

    def run_bm25_rag(self, query, chunks_directory ,top_k):

        chunks_bm25 = retrieve_with_bm25(chunks_directory, query, top_k)
        chunks_vector_db = self.query_pinecone(query, top_k)

        #combine both retrivals from pinecone and from the bm25
        combined_chunks = self._combine_chunks(chunks_bm25, chunks_vector_db)

        list_of_chunks = []
        #since chunks are stored in diff way , we move it to one list of texts
        for chunk in combined_chunks:
            if isinstance(chunk, dict):
                list_of_chunks.append(chunk["chunk_text"])
            else:
                list_of_chunks.append(chunk)

        context = "\n".join(list_of_chunks)
        full_prompt = f"Given the following context {context} what is the answer to the question: {query} (Don't repet the question in your answare)"
        response, _ = prompt_gpt(full_prompt, self.openai_client)
        return response 
 



def load_all_chunks_from_folder(folder_path):
    chunks = []
    for chunk_filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, chunk_filename), "r") as f:
            data = json.load(f)
            chunks.append(data)
    return chunks


def calculate_prompt_cost(input_tokens, output_tokens):
    return INPUT_TOKEN_PRICE * input_tokens + OUTPUT_TOKEN_PRICE * output_tokens

def get_embedding(text, model, openai_client):
    return openai_client.embeddings.create(input = [text], model=model).data[0].embedding


def prompt_gpt(prompt, openai_client):
    messages = [{"role": "user", "content": prompt}]
    response = openai_client.chat.completions.create(
        model=GPT_MODEL,  # Ensure correct model name is used
        messages=messages,
        temperature=0,
    )
    content = response.choices[0].message.content
    prompt_tokens = response.usage.prompt_tokens  # Input tokens (in)
    completion_tokens = response.usage.completion_tokens  # Output tokens (out)
    price = calculate_prompt_cost(prompt_tokens, completion_tokens)
    return content, price


def split_text_into_chunks_with_overlap(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)  # Tokenize the input text
    chunks = []
    # Loop through the tokens, creating chunks with overlap
    for i in range(0, len(tokens), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk_tokens = tokens[i:i + CHUNK_SIZE]  # Include overlap by adjusting start point
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks


def get_add_context_prompt(chunk_text, document_text):
    template = env.get_template('create_context_prompt.j2')
    data = {
        'WHOLE_DOCUMENT': document_text,  # Leave blank for default or provide a name
        'CHUNK_CONTENT': chunk_text  # You can set any score here
    }
    output = template.render(data)
    return output

def dump_docs_to_chuncs(document_dir,chunk_dir, openai_client):
    tot_price=0
    document_filenames = os.listdir(document_dir)
    for filename in tqdm(document_filenames):
        with open(f"{document_dir}/{filename}", "r", encoding="utf-8") as f:
            document_text = f.read()
        chunks = split_text_into_chunks_with_overlap(document_text)
        for idx, chunk in enumerate(chunks):
            fname = filename.split(".")[0]
            chunk_save_filename = f"{fname}_{idx}.json"
            chunk_save_path = f"{chunk_dir}/{chunk_save_filename}"
            if os.path.exists(chunk_save_path):
                continue
            prompt = get_add_context_prompt(chunk, document_text)
            context, price = prompt_gpt(prompt, openai_client)
            tot_price += price
            chunk_info = {
                "id" : f"{filename}_{int(idx)}",
                "chunk_text" : context + "\n\n" + chunk,
                "chunk_idx" : idx,
                "filename" : filename,
                "document_type": DOCUMENT_TYPE
            }
            with open(chunk_save_path, "w", encoding="utf-8") as f:
                json.dump(chunk_info, f, indent=4)



def convert_pdfs_to_text(directory_path,output_path):
    # Check if the directory exists
    if not os.path.isdir(directory_path):
        print(f"The directory {directory_path} does not exist.")
        return

    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(directory_path, filename)
            txt_path = os.path.join(output_path, filename[:-4] + '.txt')

            try:
                # Open the PDF file
                with open(pdf_path, 'rb') as pdf_file:
                    # Create a PDF reader object
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    
                    # Extract text from each page
                    text = ''
                    for page in pdf_reader.pages:
                        text += page.extract_text()

                # Write the extracted text to a new text file
                with open(txt_path, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(text)

                print(f"Successfully converted {filename} to text.")

            except Exception as e:
                print(f"Error converting {filename}: {str(e)}")

    print("Conversion process completed.")

def store_chunks_with_embeddings(chunks_directory,chunks_embedding_directory, openai_client):
    for chunk_filename in os.listdir(chunks_directory):
 # load chunk
        with open(f"{chunks_directory}/{chunk_filename}", "r", encoding="utf-8") as f:
            chunk_info = json.load(f)
            chunk_text = chunk_info["chunk_text"]
            chunk_text_embedding = get_embedding(chunk_text, TEXT_EMBEDDING_MODEL, openai_client)
            chunk_info["chunk_embedding"] = chunk_text_embedding
 # save chunk
        with open(f"{chunks_embedding_directory}/{chunk_filename}", "w", encoding="utf-8") as f:
            json.dump(chunk_info, f, indent=4)


def upload_to_pinecone(chunks_path, pinecone):
    index_name = "skybox-docs"
    index = pinecone.Index(index_name)
    # pinecone expects list of objects with: [{"id": id, "values": embedding, "metadata", metadata}]

# upload to pinecone
    for chunk_filename in os.listdir(chunks_path):
 # load chunk
        with open(f"{chunks_path}/{chunk_filename}", "r", encoding="utf-8") as f:
            chunk_info = json.load(f)
            chunk_file = chunk_info["filename"]
            chunk_idx = chunk_info["chunk_idx"]
            chunk_text = chunk_info["chunk_text"]
            chunk_text_embedding = chunk_info["chunk_embedding"]
            document_type = chunk_info["document_type"]

            metadata = {
                "filename" : chunk_file,
                "chunk_idx" : chunk_idx,
                "chunk_text" : chunk_text,
                "document_type" : document_type
            }

            data_with_metadata = [{
                "id" : chunk_filename,
                "values" : chunk_text_embedding,
                "metadata" : metadata
            }]
            index.upsert(vectors=data_with_metadata)

def retrieve_with_bm25(chunk_directory,query,top_k):

    CHUNK_PATH = chunk_directory
    #load chunks from directory
    chunks = load_all_chunks_from_folder(CHUNK_PATH)
    corpus = [chunk["chunk_text"] for chunk in chunks]

# Tokenize each document in the corpus
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus] # should store this somewhere for easy retrieval
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = word_tokenize(query.lower())
    #use the bm25 to query the chunks
    doc_scores = bm25.get_top_n(tokenized_query, corpus, n=top_k)
    return doc_scores



# Set up Jinja2 environment
file_loader = FileSystemLoader('./templates')
env = Environment(loader=file_loader)

# with redirect_stderr(open(os.devnull, "w")):
#     nltk.download('punkt', quiet=True)
#     nltk.download('punkt_tab', quiet=True)

key="PINECONE_API_KEY"
PINECONE_API_KEY=os.getenv(key)
pinecone = Pinecone(api_key=PINECONE_API_KEY)

key="OPENAI_API_KEY"
OPENAI_API_KEY=os.getenv(key)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

    
os.makedirs(CHUNKS_SAVE_PATH, exist_ok=True)
os.makedirs(TKT_FILEPATHS, exist_ok=True)
os.makedirs(CHUNKS_EMBEDDINGS_SAVE_PATH, exist_ok=True)


#this steps must be run only once for a set of documents
#first we convert pdf to txt files
convert_pdfs_to_text(PDF_FILEPATHS,TKT_FILEPATHS)

#split the txt files and dump chunk files to seperate directory
dump_docs_to_chuncs(TKT_FILEPATHS,CHUNKS_SAVE_PATH, openai_client)

#create embedding and store in a another directory
store_chunks_with_embeddings(CHUNKS_SAVE_PATH,CHUNKS_EMBEDDINGS_SAVE_PATH, openai_client)

#upload to pinecone
upload_to_pinecone(CHUNKS_EMBEDDINGS_SAVE_PATH,pinecone)

#create RAG agent and invoke the bm25 retrival
rag_agent = RagAgent("skybox-docs",pinecone,openai_client)
query = "What is the name of the centralized web access management system that Skybox will no longer support in version 14.0? : a) Okta b) SAML c) SiteMinder d) Skybox Cloud Edition"
print(rag_agent.run_bm25_rag(query, CHUNKS_SAVE_PATH, 2))

query = "What is the new URL for providing feedback in the Skybox Web UI? a) https://support.skyboxsecurity.com/s/ b) https://feedback.skyboxsecurity.com/ c) https://support.skyboxsecurity.com/s/productsurvey d) The URL has not changed."
print(rag_agent.run_bm25_rag(query, CHUNKS_SAVE_PATH, 2))

query = "Which of the following Change Manager features will NOT be available in version 13.4.110.00 but is planned for inclusion in the 14.1 release? a) Ability to edit and delete records from a CSV with multiple change requests b) Export to CSV from a ticket or a related tab c) Specify a default workflow d) All of the above."
print(rag_agent.run_bm25_rag(query, CHUNKS_SAVE_PATH, 2))

query = "Which of the following capabilities was removed from the Java Client in version 13.3? a) Network Map b) Attack Map c) Change Manager d) Access Analyzer"
print(rag_agent.run_bm25_rag(query, CHUNKS_SAVE_PATH, 2))

query = "What is the maximum number of objects supported for each dimension of an access rule context in Firewall & Network Assurance and Model Explorer? a) 1,000 b) 10,000 c) 100,000 d) 1,000,000"
print(rag_agent.run_bm25_rag(query, CHUNKS_SAVE_PATH, 2))

