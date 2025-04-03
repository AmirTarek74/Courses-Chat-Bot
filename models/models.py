import os

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_google_genai import GoogleGenerativeAI

class Models:
    def __init__(self, embedding_name, Google_API, llm_model):
        os.environ["GEMINI_API_KEY"]= Google_API
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        #genai.configure(api_key=self.gemini_api_key)

        self.llm = GoogleGenerativeAI(model=llm_model, google_api_key=self.gemini_api_key)
        self.embdding_model = HuggingFaceEmbeddings(model_name=embedding_name)