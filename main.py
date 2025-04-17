from Flows import Bots
from Agents.agents import AIChatCoordinator
import pandas as pd
import warnings
import sys 
warnings.filterwarnings("ignore", category=UserWarning)  # Hide Python warnings

sys.stdout.reconfigure(encoding='utf-8')
embedding_name = "BAAI/bge-small-en-v1.5"
google_api ="YOUR_API_KEY"
llm_name = "gemini-2.5-pro-exp-03-25"

df = pd.read_csv('data.csv', encoding='utf-8')
df.dropna(inplace=True)
columns = df.columns

texts = [f"""
{columns[0]}: {row[0]}\n{columns[1]}: {row[1]}\n{columns[2]}: {row[2]}\n{columns[3]}: {row[3]}\n{columns[4]}: {row[4]}\n{columns[5]}: {row[5]}\n
{columns[6]}: {row[6]}\n{columns[7]}: {row[7]}\n {columns[8]}: {row[8]}\n {columns[9]}: {row[9]}\n {columns[10]}: {row[10]}\n
{columns[11]}: {row[11]}\n{columns[12]}: {row[12]}\n{columns[13]}: {row[13]}\n{columns[14]}: {row[14]}\n{columns[15]}: {row[15]}\n
{columns[16]}: {row[16]}\n{columns[17]}: {row[17]}\n{columns[18]}: {row[18]}\n
""" 
    for row in df.values[0:30]]
agent = AIChatCoordinator(embedding_name=embedding_name, google_api=google_api, llm_name=llm_name,texts=texts)
query = input("Enter your query: ")
while query != 'n':
    print(agent.process_query(query))
    query = input("Enter your query: ")