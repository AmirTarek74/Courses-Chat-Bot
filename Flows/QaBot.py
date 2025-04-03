from langchain.chains import RetrievalQA, LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from models.models import Models

load_dotenv()
class QABot:
    def __init__(self, embedding_name, Google_API, llm_model,Data):
        self.models = Models(embedding_name, Google_API, llm_model)
        self.embedding = self.models.embdding_model
        self.llm = self.models.llm
        self.df = Data
    
    def setup_recommenditions(self):
        loader = DataFrameLoader(self.df, page_content_column="title")
        docs = loader.load()
        course_store = FAISS.from_documents(docs, self.embedding)

        template = """
                Recommend Courses based on {input}. Consider the user interests and skills.
                Be concise.
            """
        prompt = PromptTemplate(template=template, input_variables=["input"])
        
        return {
            "store" : course_store,
            "chain" : LLMChain(
                llm = self.llm,
                prompt= prompt
            ),
            "retriever": course_store.as_retriever()
        }

    def recommend_courses(self, query):
        flow = self.setup_recommenditions()
        similar_courses = flow["retriever"].get_relevant_documents(query)
        response = flow["chain"].invoke(
            {"input": f"Query: {query}. Similar courses: {[c.page_content for c in similar_courses]} "}
        )
        return {
            "similar courses": [c.page_content for c in similar_courses],
            "explanation": response
        }
