from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import AgentExecutor, create_openai_functions_agent
from tools import search_tool, wiki_tool, save_tool
import os

load_dotenv()


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]



llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
    streaming=False  
)

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Load tools
tools = [search_tool, wiki_tool, save_tool]

# ✅ Use openai_functions_agent (non-streaming safe)
agent = create_openai_functions_agent(
    llm=llm,
    prompt=prompt,
    tools=tools,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,  # avoids crash on formatting mismatch
)

# Run the agent
query = input("What can I help you research? ")
raw_response = agent_executor.invoke({"query": query})

try:
    structured_response = parser.parse(raw_response["output"])
    print(structured_response)
except Exception as e:
    print("Error parsing response:", e)
    print("Raw Response:", raw_response)
