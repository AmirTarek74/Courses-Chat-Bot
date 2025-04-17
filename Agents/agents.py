from Flows.Bots import RecommendBot, QABot, CareerBot
from langchain_google_genai import GoogleGenerativeAI
from crewai import Agent, Crew

import os
import json

class IntentClassifier:
    def __init__(self, Google_API, llm_model):
        os.environ["GEMINI_API_KEY"]= Google_API
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")


        self.llm = GoogleGenerativeAI(model=llm_model, google_api_key=self.gemini_api_key, verbose=False)
        self.agent = Agent(
            role="Educational Intent Classifier",
            goal="Accurately route user queries to appropriate pipelines",
            backstory="Expert in understanding educational queries and mapping them to learning systems",
            verbose=True,
            llm=self.llm,
            tools=[self.classify_intent_tool],
            memory=True
        )

    def classify_intent_tool(self, user_input: str) -> str:
        """
        This tool classifies the given user_input into one of the three categories
        - course_recommendation
        - course_qa 
        - career_coaching
        
        The tool will return the category name as a string
        """
        classification_prompt = f"""
        Classify this query into one of these categories:
        - course_recommendation
        - course_qa 
        - career_coaching
        
        Query: {user_input}
        
        Respond ONLY with the category name.
        """
        
        response = self.llm.invoke(classification_prompt)
        return response.content.strip().lower()

    def classify(self, user_input: str) -> str:
        """
        Classify the provided user input into one of the predefined categories.

        Args:
            user_input (str): The input query from the user that needs to be classified.

        Returns:
            str: The category name which can be one of 'course_recommendation', 'course_qa', or 'career_coaching'.
        """

        return self.agent.work(user_input)
    
class ContextHandler:
    def __init__(self, Google_API, llm_model):
        os.environ["GEMINI_API_KEY"]= Google_API
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")


        self.llm = GoogleGenerativeAI(model=llm_model, google_api_key=self.gemini_api_key, verbose=False)
        self.agent = Agent(
            role="Educational Response Validator",
            goal="Ensure responses are appropriate, safe, and contextually relevant",
            backstory="Quality assurance expert for educational content",
            verbose=True,
            llm=self.llm,
            tools=[self.validate_response_tool],
            memory=True
        )

    def validate_response_tool(self, response: str) -> dict:
        """
        Analyze the given chatbot response for:
        1. Appropriate educational content
        2. Safety (no PII, harmful content)
        3. Relevance to original query
        4. Clarity and readability

        Return a dictionary in the following format:
        {
            "valid": boolean,
            "issues": list[str],
            "sanitized_response": string
        }
        """
        
        
        validation_prompt = f"""
        Analyze this chatbot response:
        {response}
        
        Check for:
        1. Appropriate educational content
        2. Safety (no PII, harmful content)
        3. Relevance to original query
        4. Clarity and readability
        
        Return JSON format:
        {{
            "valid": boolean,
            "issues": list[str],
            "sanitized_response": string
        }}
        """
        
        validation = self.llm.invoke(validation_prompt)
        return json.loads(validation.content)

    def validate(self, response: str) -> dict:
        """
        Run the validation tool on the given chatbot response.
        
        Args:
            response: str, the chatbot response to validate
        
        Returns:
            dict, containing "valid", "issues", and "sanitized_response".
        """
        
        return self.agent.work(response)
    

class AIChatCoordinator:
    def __init__(self, embedding_name, google_api, llm_name,texts):
        self.intent_classifier = IntentClassifier(Google_API=google_api, llm_model=llm_name)
        self.context_handler = ContextHandler(Google_API=google_api, llm_model=llm_name)

        self.recommend = RecommendBot(embedding_name, google_api, llm_name,texts).recommend_courses()
        self.qabot = QABot(embedding_name, google_api, llm_name,texts).answer_course_question()
        self.careerbot = CareerBot(embedding_name, google_api, llm_name,texts).answer_career_question
        
        self.chains = {
            'course_recommendation': self.recommend,
            'course_qa': self.qabot,
            'career_coaching': self.careerbot
        }

    def process_query(self, user_input: str) -> str:
        # Step 1: Intent classification
        
        """
        Process a user query by running it through the intent classifier, the appropriate
        chain (e.g. recommend, QA, or career coaching), and then the context validator.
        
        Args:
            user_input: str, the user's query
        
        Returns:
            str, the validated response from the chatbot
        """
        intent = self.intent_classifier.classify(user_input)
        
        # Step 2: Execute appropriate chain
        chain = self.chains.get(intent, self.qabot)  # Default to QA
        raw_response = chain.invoke({"query": user_input})
        
        # Step 3: Context validation
        validation = self.context_handler.validate(raw_response['result'])
        
        return validation.get('sanitized_response', "Sorry, I couldn't process that request.")