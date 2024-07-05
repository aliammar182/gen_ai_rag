import os

from langchain_community.document_loaders import  PDFMinerLoader, Docx2txtLoader


ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Define the folder for storing database
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS"


# Can be changed to a specific number
INGEST_THREADS = os.cpu_count() or 8


# https://python.langchain.com/en/latest/_modules/langchain/document_loaders/excel.html#UnstructuredExcelLoader
DOCUMENT_MAP = {
    ".pdf": PDFMinerLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}

