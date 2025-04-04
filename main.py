from Flows import Bots
import pandas as pd

embedding_name = "BAAI/bge-small-en-v1.5"
google_api ="AIzaSyCvgEfXtii7hpQkHXtTyYCRmAfMoktFrPQ"
llm_name = "gemini-2.5-pro-exp-03-25"

data = {
    "course_id": [101, 102, 103],
    "title": ["Data Science 101", "Python Fundamentals", "Machine Learning Advanced"],
    "description": [
        "Introduction to data science concepts",
        "Basic Python programming course",
        "Advanced ML techniques and implementations"
    ],
    "duration": ["8 weeks", "4 weeks", "10 weeks"],
    "career_paths": [
        "Data Analyst, Business Intelligence",
        "Python Developer, Automation Engineer",
        "ML Engineer, AI Researcher"
    ],
    "prerequisites": ["None", "Basic programming", "Linear algebra"]
}

df = pd.DataFrame(data)
texts = [f"Course ID: {row[0]}\nTitle: {row[1]}\nDescription: {row[2]}\nDuration: {row[3]}.\tCareer Paths: {row[4]}\nPrerequisites: {row[5]}\n" 
         for row in df.values]


bot = Bots.QABot(embedding_name, google_api, llm_name,texts)
query = input("Enter your query: ")
while query != 'n':
    print(bot.answer_course_question(query))
    query = input("Enter your query: ")