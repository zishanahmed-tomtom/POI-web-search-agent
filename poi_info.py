from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from langchain_community.llms import Ollama,OpenAIChat
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from crewai_tools import WebsiteSearchTool,SerperDevTool

search_tool = SerperDevTool()
web_tool = WebsiteSearchTool()

load_dotenv()

#llm = Ollama(model="llama3:latest", verbose=True)
#llm = ChatOpenAI(temperature=0.1, model="gpt-4.0-mini")
llm = OpenAIChat(temperature=0.1, model="gpt-4.0-mini")


def kickoffcrew(POI, address):
    researcher = Agent(
        role="Internet Research",
        goal=f"Perform research on {POI} at address {address}, & find website url ,email id and opening hours of {POI}.It is alright if some info is missing",
        verbose=True,
        llm=llm,
        max_iterations=2,
        backstory="""You are an expert Internet Researcher Who knows how to search the internet for detailed content 
        on {POI} at address {address}."""
    )

    task_search = Task(
        description="""Search for  the details about the {POI} at address {address}.Your final answer should contain 
        the website url ,email id and opening hours of POI.It is alright if some info is missing. This content should pzth
        be well organized, and should be very easy to read""",
        expected_output='A brief info about website url,email id and opening hours of {POI} at address {address} in '
                        'markdown format. It is ok if some info is missing',

        tools=[search_tool, web_tool],
        agent=researcher)

    crew = Crew(
        agents=[researcher],
        tasks=[task_search],
        verbose=True,
        process=Process.sequential)

    result = crew.kickoff()
    return result



result = kickoffcrew("Starbucks", "Sugarbush drive, 1055")
print(result)

