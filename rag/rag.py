import os
import shutil
import time

from decouple import config

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

# Função para esvaziar a pasta sem deletá-la
def esvaziar_pasta(pasta):
    if os.path.exists(pasta):
        for item in os.listdir(pasta):
            item_path = os.path.join(pasta, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"⚠️ Não foi possível remover {item_path}: {e}")



os.environ['HUGGINGFACE_API_KEY'] = config('HUGGINGFACE_API_KEY')


if __name__ == '__main__':
    file_path = '/app/rag/data/USF_BOM_JESUS_2-ATALIZADO-20-04-2025.pdf'
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(
        documents=docs,
    )

    print(f"🔍 Total de chunks gerados: {len(chunks)}")

    persist_directory = '/app/chroma_data'

   # Esvaziar pasta se já existir
    esvaziar_pasta(persist_directory)
    print(f"🧹 Conteúdo de {persist_directory} limpo com sucesso.")


    # Criar novo banco com embeddings atualizados
    try:
        print("📡 Inicializando HuggingFaceEmbeddings...")
        embedding = HuggingFaceEmbeddings()
        print("✅ Embedding inicializado.")
    except Exception as e:
        print(f"❌ Erro ao inicializar embeddings: {e}")
        exit()
    
    # embedding = HuggingFaceEmbeddings()

    try:
        print("📦 Criando Chroma vector store...")
        vector_store = Chroma(
            embedding_function=embedding,
            persist_directory=persist_directory,
        )
        print("✅ Chroma inicializado.")
    except Exception as e:
        print(f"❌ Erro ao inicializar Chroma: {e}")
        exit()

    # vector_store = Chroma(
    #     embedding_function=embedding,
    #     persist_directory=persist_directory,
    # )
    print("📥 Adicionando documentos ao vetor store...")

    vector_store.add_documents(documents=chunks)

    print("💾 Persistindo banco vetorial...")
    print("✅ Banco vetorial recriado com sucesso.")

    #  # Apagar o banco antigo, se existir
    # if os.path.exists(persist_directory):
    #     shutil.rmtree(persist_directory)
    #     print(f"🧹 Banco vetorial anterior removido: {persist_directory}")

    # # Criar novo banco com embeddings atualizados
    # embedding = HuggingFaceEmbeddings()
    # vector_store = Chroma(
    #     embedding_function=embedding,
    #     persist_directory=persist_directory,
    # )
    # vector_store.add_documents(documents=chunks)
    # vector_store.persist()

    # print("✅ Banco vetorial recriado com sucesso.")