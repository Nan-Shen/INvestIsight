from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import SerperDevTool, FirecrawlScrapeWebsiteTool
from crewai.tools import tool
from typing import List, Type
from pydantic import BaseModel, Field

#from tools.custom_tool import ScrapeTool

from dotenv import load_dotenv
from pathlib import Path
dotenv_path = Path('../../keys/.env')
load_dotenv(dotenv_path=dotenv_path)
    
class InsightOutline(BaseModel):
    heading: str = Field( 
        description="The title of the outline section, summarize the key points of this section."
    )
    bullet_points: List[str] = Field(
        description="A list of key points summarizing this section."
    )
    
class Outline(BaseModel):
    sections: List[InsightOutline] = Field(
        description="A list of outline sections defining the article structure."
    )

class Section(BaseModel):
    """Section of the report"""
    title: str = Field(
        description="Markdown level-2 heading (e.g., '## Section Title')."
    )
    content: str = Field(
        description="Markdown-formatted prose for this section."
    )

class Report(BaseModel):
    """Full report"""
    sections: List[Section] = Field(
        description="Ordered list of the report sections with their titles and content."
    )


# @tool('search_tool')
# def search(search_query: str):
#     """Search the web for information on a given topic"""
#     response = SerperDevTool().run(search_query)
#     return response

# @tool('scrape_tool')
# def scrape(url: str):
#     """Search the web for information on a given topic"""
#     response = ScrapeTool().run(url)
#     return response

# from crewai.tools import BaseTool
# from pydantic import BaseModel, Field

# class SearchInput(BaseModel):
#     """Input schema for search Tool."""
#     search_query: str = Field(description="The search query")

# class search(BaseTool):
#     name: str = "search tool"
#     description: str = "Searches the internet using SerperDevTool.",
#     args_schema: Type[BaseModel] = SearchInput
#     api_key: str = os.getenv("SERPER_API_KEY")

#     def _run(self, search_query: str) -> str:
#         app = SerperDevTool()
#         res = app.run(search_query=search_query)
#         return res

@CrewBase
class PlanCrew:
    """Plan and validate Crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self):
        # self.search_tool = search
        self.scrape_tool = FirecrawlScrapeWebsiteTool()
        self.search_tool = SerperDevTool()
        # self.llm = LLM(
        #             model="ollama/llama3.2:1b",
        #             base_url="http://localhost:11434"
        # )
        self.llm = 'gpt-4o-mini'
        # ollama tool calling bug fix: https://medium.com/google-cloud/building-ai-agents-with-google-adk-gemma-3-and-mcp-tools-28763a8f3c62          
    @agent
    def plan_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["plan_agent"],
            llm=self.llm,
            allow_delegation=False,
            tools=[self.search_tool, self.scrape_tool],
            function_calling_llm=self.llm
        )
        
    @task
    def plan_task(self) -> Task:
        return Task(
            config=self.tasks_config["plan_task"],
            output_pydantic=Outline,
            tools=[self.search_tool],
            human_input=True
        )
        
    @crew
    def crew(self) -> Crew:
        """Creates the Research Crew"""
        return Crew(
            agents=self.agents, 
            tasks=self.tasks, 
            #process=Process.sequential,
            verbose=True,
        )
