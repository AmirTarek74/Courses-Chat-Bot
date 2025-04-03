from Flows import QaBot
import pandas as pd

embedding_name = "BAAI/bge-small-en-v1.5"
google_api ="Your API here"
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

qabot = QaBot.QABot(embedding_name, google_api, llm_name, df)
print(qabot.recommend_courses("I want to learn data science"))