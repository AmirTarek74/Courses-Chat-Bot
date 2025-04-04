from langchain.chains import RetrievalQA, LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from models.models import Models

load_dotenv()
class RecommendBot:
    def __init__(self, embedding_name, Google_API, llm_model,text):
        self.models = Models(embedding_name, Google_API, llm_model)
        self.embedding = self.models.embdding_model
        self.llm = self.models.llm
        self.text = text
    
    def recommend_courses(self, query):
        
        course_store = FAISS.from_documents(self.text, self.embedding)

        template = """<|user|>
                Use the following pieces of context to recommend courses to the user based on the input.
                Describe each recommended course details, if found.
                If you don't know the answer, just say that you don't know,
                Relvant information:
                {context}
                Input: {question}<|end|>
                <|assistant|>
                """
        prompt = PromptTemplate(template=template, input_variables=["input", "context"])
    
        rag = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=course_store.as_retriever(search_kwargs={'k': 2}),
            chain_type_kwargs={"prompt": prompt},  
            return_source_documents=True,
            verbose=True
        )
        response = rag.invoke(query)
        return response['result']

    


class QABot:
    def __init__(self, embedding_name, Google_API, llm_model,text):
        self.models = Models(embedding_name, Google_API, llm_model)
        self.embedding = self.models.embdding_model
        self.llm = self.models.llm
        self.text = text
    
    def answer_course_question(self, quesiton):
        
        qa_store = FAISS.from_texts(self.text, self.embedding)
        template = """
                <|user|>
                Use the following pieces of context to answer the question at the end about the given course name.
                If you don't know the answer, just say that you don't know,
                Relvant information:
                {context}
                Question: {question}<|end|>
                <|assistant|>
            """
        prompt = PromptTemplate(template=template, input_variables=["input", "context"])
        rag = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=qa_store.as_retriever(search_kwargs={'k': 2}),
            chain_type_kwargs={"prompt": prompt},  
            return_source_documents=True,
            verbose=True
        )
        response = rag.invoke(quesiton)
        return response['result']
    
   
