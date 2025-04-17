from langchain.chains import RetrievalQA, LLMChain, SequentialChain, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from models.models import Models

load_dotenv()
class RecommendBot:
    def __init__(self, embedding_name, Google_API, llm_model,text):
        self.models = Models(embedding_name, Google_API, llm_model)
        self.embedding = self.models.embdding_model
        self.llm = self.models.llm
        self.text = text
    
    def recommend_courses(self):
        
        course_store = FAISS.from_texts(self.text, self.embedding)

        template = """<|user|>
                You are an expert educational advisor. A user has the following learning objective:
                "{question}"

                Based on our course offerings, here is some summarized information:
                {context}

                Please recommend the top courses that match this query along with key details for each course (e.g., course content, duration, prerequisites). Present the information in clear bullet points.
                If unsure, say you don't know. <|end|>
                <|assistant|>
                """
        prompt = PromptTemplate(template=template, input_variables=["question", "context"])
    
        rag = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=course_store.as_retriever(search_kwargs={'k': 2}),
            chain_type_kwargs={"prompt": prompt},  
            return_source_documents=True,
            verbose=True
        )
        #response = rag.invoke({"query": query})
        return rag

    


class QABot:
    def __init__(self, embedding_name, Google_API, llm_model,text):
        self.models = Models(embedding_name, Google_API, llm_model)
        self.embedding = self.models.embdding_model
        self.llm = self.models.llm
        self.text = text
    
    def answer_course_question(self):
        
        """
        Answer a user's course-related question based on provided course details.

        This method utilizes a FAISS-based retrieval system to find relevant course 
        information and uses a language model to generate a detailed response. It 
        employs a predefined prompt template to ensure that the response is clear 
        and based on the context provided by the course data. If the system is 
        unable to generate a confident answer, it will indicate uncertainty.

        Args:
            quesiton (str): The question posed by the user regarding the course.

        Returns:
            str: A detailed answer to the user's question, or an indication of 
                uncertainty if the question cannot be confidently answered.
        """

        qa_store = FAISS.from_texts(self.text, self.embedding)
        template = """
                <|user|>
                You are an educational expert with in-depth knowledge of the course details provided below:

                Course Context:
                {context}

                User Question:
                {question}

                Provide a clear and detailed answer based on the above course information. If unsure, say you don't know..<|end|>
                <|assistant|>
            """
        prompt = PromptTemplate(template=template, input_variables=["question", "context"])
        rag = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=qa_store.as_retriever(search_kwargs={'k': 2}),
            chain_type_kwargs={"prompt": prompt},  
            return_source_documents=True,
            verbose=True
        )
        #response = rag.invoke({"query": quesiton})
        return rag
    
   
class CareerBot:
    def __init__(self, embedding_name, Google_API, llm_model, text):
        self.models = Models(embedding_name, Google_API, llm_model)
        self.embedding = self.models.embdding_model  
        self.llm = self.models.llm
        self.text = text
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            #input_key='question',  # Align with chain's input
            return_messages=True
        )

    def answer_career_question(self):
        """
        Answer a user's question about their career or further learning after completing a course.

        This method utilizes a FAISS-based retrieval system to find relevant course 
        information and uses a language model to generate a detailed response. It 
        employs a predefined prompt template to ensure that the response is clear 
        and based on the context provided by the course data. The method also makes
        use of a memory system to keep track of the conversation history and allow
        for more informed and contextual responses.

        Args:
            question (str): The question posed by the user regarding their career or further learning.

        Returns:
            str: A detailed answer to the user's question, or an indication of uncertainty if the question cannot be confidently answered.
        """
        qa_store = FAISS.from_texts(self.text, self.embedding)
        
        template = """
            <|user|>
            Chat History: {chat_history}
            You have a strong understanding of the skills and knowledge provided by the following course:
            {context}

            The user is seeking guidance on the next steps in their career or further learning after completing the course. Here is the question:
            {question}

            Please provide detailed, actionable advice, including additional courses, skills to develop, or career paths worth considering.
            If unsure, say you don't know.<|end|>
            <|assistant|>
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=["chat_history", "context", "question"]  # Fixed input vars
        )
        
        
        rag = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=qa_store.as_retriever(search_kwargs={'k': 2}),
            chain_type_kwargs={"prompt": prompt},  
            return_source_documents=True,
            verbose=True
        )
      
        #response = rag.invoke({"question": question})
        return rag