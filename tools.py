from langchain_chroma import Chroma
from pathlib import Path
from langchain_openai import OpenAIEmbeddings

openai_embeddings = OpenAIEmbeddings()



# telegram message tool
tg_folder_path = './tools_embeddings/tg_embeddings'
tg_db = Chroma(persist_directory=tg_folder_path, embedding_function=openai_embeddings)

def get_tg_retriever(search_type = "similarity_score_threshold", k = 6, threshold = .8):
    return tg_db.as_retriever(search_type=search_type, search_kwargs={'k': k, "score_threshold": threshold})



# website tool
website_path = './tools_embeddings/vstup_website_embeddings'
website_db = Chroma(persist_directory=website_path, embedding_function=openai_embeddings)

def get_website_retriever(search_type = "similarity_score_threshold", k = 2, threshold = .6):
    return website_db.as_retriever(search_type=search_type, search_kwargs={'k': k, "score_threshold": threshold})



#passing scores tool
passing_score_path = "./tools_embeddings/passing_score_embeddings"
passing_score_db = Chroma(persist_directory=passing_score_path, embedding_function=openai_embeddings)

def get_passing_score_retriever(search_type = "similarity_score_threshold", k = 2, threshold = .6):
    return passing_score_db.as_retriever(search_type=search_type, search_kwargs={'k': k, "score_threshold": threshold})



#disciplines tool
disciplines_path = "./tools_embeddings/disciplines_embeddings"
disciplines_db = Chroma(persist_directory=disciplines_path, embedding_function=openai_embeddings)

def get_disciplines_retriever(search_type = "similarity_score_threshold", k = 4, threshold = .8):
    return disciplines_db.as_retriever(search_type=search_type, search_kwargs={'k': k, "score_threshold": threshold})

# print(get_disciplines_retriever().invoke("компютерні науки"))