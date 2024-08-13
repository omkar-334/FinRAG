import io
import os
import tarfile
from datetime import datetime

import requests
from dotenv import dotenv_values
from huggingface_hub import HfApi, create_repo
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_parse import LlamaParse
from mthrottle import Throttle
from qdrant_client import QdrantClient
from serpapi import GoogleSearch

apidict = dict(dotenv_values())
throttle_config = {"lookup": {"rps": 15}, "default": {"rps": 8}}
th = Throttle(throttle_config, 15)

instruction = """
                The provided document is a press release filed by the company {} for the year {} and quarter {} with the Securities and Exchange Commission (SEC).
                This form provides detailed financial information about the company's performance for a specific quarter.
                It includes unaudited financial statements, management discussion and analysis, and other relevant disclosures required by the SEC.
                It contains many tables. Try to be precise while answering the questions.
                """


def serp_request(company: str, year: int = None, quarter: int = None):
    if not year:
        year = datetime.now().year
    if not quarter:
        quarter = (datetime.now().month) // 3

    params = {"q": f"{company} press release {year} Q{quarter} filetype:pdf", "num": 1, "hl": "en", "source": "python", "serp_api_key": apidict["SERPAPI_KEY"]}
    search = GoogleSearch(params).get_dict()
    link = search["organic_results"][0]["link"]

    return link


def download(url, filename):
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/118.0", "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.5"})

    th.check()
    try:
        with session.get(url, stream=True, timeout=10) as r:
            r.raise_for_status()
            with open(filename, mode="wb") as f:
                for chunk in r.iter_content(chunk_size=1000000):
                    f.write(chunk)
    except:
        pass
    session.close()


def parse_pdf(filename, instruction):
    parser = LlamaParse(
        api_key=apidict["LLAMAPARSE_KEY"],
        result_type="markdown",
        parsing_instruction=instruction,
    )
    mdparser = MarkdownNodeParser()

    documents = parser.load_data(filename)
    filename = filename.replace("pdf", "md")
    with open(filename, "w") as f:
        for doc in documents:
            f.write(doc.text + "\n")
    nodes = mdparser.get_nodes_from_documents(documents)
    return nodes


def collection_exists(collection):
    url = f"http://localhost:6333/collections/{collection}/exists"
    response = requests.get(url)
    return response.json()["result"]["exists"]


def upload_nodes(nodes, collection_name):
    client = QdrantClient(url="http://localhost:6333")
    embed_model = CohereEmbedding(
        api_key=apidict["COHERE_KEY"],
        model_name="embed-english-light-v3.0",
        input_type="search_document",
    )
    embed_model = HuggingFaceEmbedding(model_name="gaianet/Nomic-embed-text-v1.5-Embedding-GGUF")
    contents = [node.get_content(metadata_mode="all") for node in nodes]
    embeddings = embed_model.get_text_embedding_batch(contents)
    length = len(nodes)
    outnodes = []
    for idx in range(length):
        embed = embeddings[idx]
        if embed:
            node = nodes[idx]
            node.embedding = embed
            outnodes.append(node)

    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(outnodes, storage_context=storage_context, embed_model=embed_model)
    return index


def retrieve_snapshot(collection_name, tar=False):
    client = QdrantClient(url="http://localhost:6333")
    snap = client.create_snapshot(collection_name=collection_name)
    url = f"http://localhost:6333/collections/{collection_name}/snapshots/{snap.name}"
    response = requests.get(url)

    file_name = f"{collection_name}.snapshot"

    if tar:
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w:gz") as tar:
            tarinfo = tarfile.TarInfo(name=file_name)
            tarinfo.size = len(response.content)
            tar.addfile(tarinfo, io.BytesIO(response.content))
        tar_path = os.path.join("snapshots", f"{collection_name}.tar.gz")
        with open(tar_path, "wb") as f:
            f.write(tar_stream.getvalue())
    else:
        path = os.path.join("snapshots", collection_name + ".snapshot")
        with open(path, "wb") as f:
            response.raise_for_status()
            f.write(response.content)
    return tar_path


def upload_hf(filepath, filename, hfpath, new=True):
    if new:
        create_repo(hfpath)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=filepath,
        path_in_repo=filename,
        repo_id=hfpath,
        repo_type="dataset",
    )
    return f"https://huggingface.co/{hfpath}"


def fetch_data(company: str, year: int = None, quarter: int = None):
    collection_name = f"{company}_{year}Q{quarter}"
    file_name = os.path.join("data", f"{collection_name}.pdf")
    if os.path.isfile(file_name):
        return "Data already exists in cache."

    file_url = serp_request(company, year, quarter)
    download(file_url, file_name)
    if not os.path.isfile(file_name):
        return "Error. File not found"

    nodes = parse_pdf(file_name, instruction.format(company, year, quarter))
    upload_nodes(nodes, collection_name)
    retrieve_snapshot(collection_name)
    return "Data fetched, embedded and uploaded"
