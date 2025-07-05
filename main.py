import os
import torch
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI  # ✅ Replacing LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# ✅ Set your OpenAI API key here (or use environment variable)
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# ✅ Set device for embeddings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load PDF and split text
loader = PyPDFLoader(file_path=r"Sachin.pdf")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
text_chunks = text_splitter.split_documents(data)

# ✅ Initialize OpenAI LLM instead of LlamaCpp
llm_answer_gen = ChatOpenAI(
    model_name="gpt-3.5-turbo",  # or "gpt-4"
    temperature=0.75
)

# ✅ Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)

# ✅ Create vector DB
vector_store = Chroma.from_documents(text_chunks, embeddings)

# ✅ Build the conversational chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
answer_gen_chain = ConversationalRetrievalChain.from_llm(
    llm=llm_answer_gen,
    retriever=vector_store.as_retriever(),
    memory=memory
)

# ✅ Main chat loop
while True:
    user_input = input("Enter a question (or type 'q' to quit): ")
    if user_input.lower() == 'q':
        break

    answers = answer_gen_chain.run({"question": user_input})
    print("Answer:", answers)
