import os
import csv
from datetime import datetime

import boto3
from langchain_community.embeddings.bedrock import BedrockEmbeddings

# from langchain.embeddings import HuggingFaceInstructEmbeddings
# from langchain.embeddings import HuggingFaceBgeEmbeddings
# from langchain.embeddings import HuggingFaceEmbeddings


# def log_to_csv(question, answer):

#     log_dir, log_file = "local_chat_history", "qa_log.csv"
#     # Ensure log directory exists, create if not
#     if not os.path.exists(log_dir):
#         os.makedirs(log_dir)

#     # Construct the full file path
#     log_path = os.path.join(log_dir, log_file)

#     # Check if file exists, if not create and write headers
#     if not os.path.isfile(log_path):
#         with open(log_path, mode="w", newline="", encoding="utf-8") as file:
#             writer = csv.writer(file)
#             writer.writerow(["timestamp", "question", "answer"])

#     # Append the log entry
#     with open(log_path, mode="a", newline="", encoding="utf-8") as file:
#         writer = csv.writer(file)
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         writer.writerow([timestamp, question, answer])



def get_embeddings(device_type=None):
    
        #  aws_profile = 'everbanega'
        #  boto3.setup_default_session(profile_name=aws_profile)
         bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)
         titan_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",
                                     client=bedrock)

         return titan_embeddings
