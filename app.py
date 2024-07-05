from flask import Flask, render_template, request, jsonify
import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
import traceback
import boto3
import time
from pinecone import Pinecone, ServerlessSpec
from langchain.llms import HuggingFaceHub
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms.bedrock import Bedrock
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from ingest import main as process_docs, load_documents_from_s3
from prompt_template import get_prompt_template
from openai import OpenAI

load_dotenv()

pinecone_api_key = os.environ.get('PINECONE_API_KEY')
pc = Pinecone(api_key=pinecone_api_key)

# aws_profile = 'everbanega'
# boto3.setup_default_session(profile_name=aws_profile)

bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)

titan_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

prompt, memory = get_prompt_template(passed_prompt=None)

def load_model(model_name=None, model_kwargs=None):
    if model_name is None or model_name == "mistral-7b":
        kwargs = model_kwargs or {
            "max_tokens": 200,
            "temperature": 0.6}
        llm = Bedrock(model_id="mistral.mistral-7b-instruct-v0:2", client=bedrock, model_kwargs=kwargs)
    else:
        llm = ChatOpenAI(
            model=model_name,
            temperature=model_kwargs.get("temperature", 0.6),
            max_tokens=model_kwargs.get("max_tokens", 200),
            timeout=None,
            max_retries=2,
            api_key=os.environ.get('OPENAI_API_KEY')
        )
    return llm

llm = load_model()

INDEX_TRACKER_FILE = "folder_index_mapping.json"

def load_index_tracker():
    if os.path.exists(INDEX_TRACKER_FILE):
        with open(INDEX_TRACKER_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_index_tracker(index_tracker):
    with open(INDEX_TRACKER_FILE, 'w') as f:
        json.dump(index_tracker, f)

def create_pinecone_index(index_name):
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

app = Flask(__name__)

chats = {}
chat_counter = 1
vector_store = None
llm_chain = None
current_index_name = None


def get_model_ids():
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    models = client.models.list()
    model_ids = [model.id for model in models.data if 'gpt' in model.id.lower()]
    return model_ids

def get_response(llm, vector_store):
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        ),
        chain_type_kwargs={"prompt": prompt, "memory": memory},
        return_source_documents=False,
        verbose=True
    )
    return retrieval_qa

def get_folders():
    s3 = boto3.client('s3')
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    
    response = s3.list_objects_v2(Bucket=bucket_name, Delimiter='/')
    folders = [prefix['Prefix'].rstrip('/') for prefix in response.get('CommonPrefixes', [])]

    return folders

@app.route('/get_s3_folders', methods=['GET'])
def get_s3_folders():
    try:
        folders = get_folders()
        return jsonify({'status': 'success', 'folders': folders})
    except Exception as e:
        print(f"Error fetching folders from S3: {e}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/')
def index_view():
    return render_template('index.html')

@app.route('/settings')
def settings_view():
    model_ids = get_model_ids()
    return render_template('settings.html', model_ids=model_ids)

@app.route('/send_message', methods=['POST'])
def send_message():
    global llm_chain, vector_store, current_index_name
    try:
        data = request.json
        chat_id = data.get('chat_id')
        message = data.get('message')
        selected_folder = data.get('folder')
        index_name = re.sub(r'[^a-z0-9-]+', '-', selected_folder.lower())

        if chat_id not in chats:
            chats[chat_id] = {
                'id': chat_id,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'messages': []
            }
        print('indexes are', pc.list_indexes().names())
        print('index name is', index_name)
        if index_name not in pc.list_indexes().names():
            return jsonify({'status': 'error', 'message': 'Please update the knowledge base to start.'}), 400
        print('llm is', llm)
        if vector_store is None or current_index_name != index_name:
            vector_store = PineconeVectorStore(index_name=index_name, embedding=titan_embeddings)
            llm_chain = get_response(llm, vector_store)
            current_index_name = index_name

        full_response = llm_chain.invoke(message)
        full_response = full_response['result'].strip()
        print('full response is', full_response)

        match = re.search(r'Answer:(.*)', full_response, re.DOTALL)
        bot_response = full_response

        chats[chat_id]['messages'].append({'user': message, 'bot': bot_response})

        print('bot_response is', bot_response)

        return jsonify({'user': message, 'bot': bot_response})
    except Exception as e:
        print(f"Error in send_message: {e}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/new_chat', methods=['POST'])
def new_chat():
    global chat_counter, memory, llm_chain, vector_store
    chat_id = chat_counter
    chat_counter += 1
    chats[chat_id] = {
        'id': chat_id,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'messages': []
    }

    memory = get_prompt_template()[1]
    llm_chain = get_response(llm, vector_store)

    return jsonify(chats[chat_id])

@app.route('/get_chats', methods=['GET'])
def get_chats():
    return jsonify(list(chats.values()))

@app.route('/get_chat/<int:chat_id>', methods=['GET'])
def get_chat(chat_id):
    global llm_chain, vector_store
    chat = chats.get(chat_id)
    if not chat:
        return jsonify({'error': 'Chat not found'}), 404

    for msg in chat['messages']:
        if 'user' in msg and 'bot' in msg:
            memory.save_context({"question": msg['user']}, {"result": msg['bot']})

    llm_chain = get_response(llm, vector_store)

    return jsonify(chat)

@app.route('/process_folder', methods=['POST'])
def process_folder():
    global vector_store, llm_chain, memory, current_index_name
    try:
        data = request.json
        folder = data.get('folder')
        index_name = re.sub(r'[^a-z0-9-]+', '-', folder.lower())

        index_tracker = load_index_tracker()

        if folder in index_tracker and index_tracker[folder] in pc.list_indexes().names():
            return jsonify({
                'status': 'existing',
                'message': 'Index already exists. Do you want to update the knowledge base or switch to the existing one?',
                'options': ['update', 'switch']
            })

        create_pinecone_index(index_name)
        index_tracker[folder] = index_name
        save_index_tracker(index_tracker)

        vector_store = PineconeVectorStore(index_name=index_name, embedding=titan_embeddings)
        memory = get_prompt_template()[1]
        llm_chain = get_response(llm, vector_store)
        current_index_name = index_name

        bucket_name = os.environ.get('S3_BUCKET_NAME')
        documents = load_documents_from_s3(bucket_name, folder)
        
        process_docs('cpu', documents)

        return jsonify({'status': 'success', 'message': f'Processed documents in folder: {folder}'})
    except Exception as e:
        print(f"Error processing folder: {e}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/update_knowledge_base', methods=['POST'])
def update_knowledge_base():
    global vector_store, llm_chain, memory, current_index_name
    try:
        data = request.json
        folder = data.get('folder')
        index_name = re.sub(r'[^a-z0-9-]+', '-', folder.lower())

        vector_store = PineconeVectorStore(index_name=index_name, embedding=titan_embeddings)
        memory = get_prompt_template()[1]
        llm_chain = get_response(llm, vector_store)
        current_index_name = index_name

        bucket_name = os.environ.get('S3_BUCKET_NAME')
        documents = load_documents_from_s3(bucket_name, folder)
        
        process_docs('cpu', documents)

        return jsonify({'status': 'success', 'message': f'Updated knowledge base with documents in folder: {folder}'})
    except Exception as e:
        print(f"Error updating knowledge base: {e}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/switch_knowledge_base', methods=['POST'])
def switch_knowledge_base():
    global vector_store, llm_chain, memory, current_index_name
    try:
        data = request.json
        folder = data.get('folder')
        index_name = re.sub(r'[^a-z0-9-]+', '-', folder.lower())

        vector_store = PineconeVectorStore(index_name=index_name, embedding=titan_embeddings)
        memory = get_prompt_template()[1]
        llm_chain = get_response(llm, vector_store)
        current_index_name = index_name

        return jsonify({'status': 'success', 'message': f'Switched to knowledge base: {folder}'})
    except Exception as e:
        print(f"Error switching knowledge base: {e}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/save_memory', methods=['POST'])
def save_memory():
    global llm_chain, vector_store
    try:
        data = request.json
        input_text = data.get('input')
        output_text = data.get('output')
        print(f"Received request to save memory with input: {input_text} and output: {output_text}")

        memory.save_context({"question": input_text}, {"result": output_text})
        llm_chain = get_response(llm, vector_store)

        return jsonify({'status': 'success', 'message': 'Memory saved'})
    except Exception as e:
        print(f"Error saving memory: {e}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/save_settings', methods=['POST'])
def save_settings():
    global llm, llm_chain, vector_store, current_index_name
    try:
        data = request.json
        model_name = data.get('model')
        model_kwargs = {
            "max_tokens": int(data.get('max_tokens', 200)),
            "temperature": float(data.get('temperature', 0.6))
        }
        passed_prompt = data.get('passed_prompt')
        print('passed prompt is', passed_prompt)

        if model_name == "openai":
            openai_model = data.get('openai_model')
            llm = load_model(model_name=openai_model, model_kwargs=model_kwargs)
        else:
            llm = load_model(model_name=model_name, model_kwargs=model_kwargs)

        prompt, memory = get_prompt_template(passed_prompt=passed_prompt)

        if current_index_name is None:
            return jsonify({'status': 'error', 'message': 'Pinecone index name is not set. Please select a folder to process first.'}), 400

        vector_store = PineconeVectorStore(index_name=current_index_name, embedding=titan_embeddings)
        llm_chain = get_response(llm, vector_store)

        return jsonify({'status': 'success', 'message': 'Settings updated'})
    except Exception as e:
        print(f"Error saving settings: {e}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
