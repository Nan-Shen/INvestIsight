from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import SerperDevTool, FirecrawlScrapeWebsiteTool
from crewai.tools import tool
from typing import List, Type
from pydantic import BaseModel, Field
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


@CrewBase
class PlanCrew:
    """Plan and validate Crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self):
        self.scrape_tool = FirecrawlScrapeWebsiteTool()
        self.search_tool = SerperDevTool()
        # self.llm = LLM(
        #             model="ollama/llama3.2:1b",
        #             base_url="http://localhost:11434"
        # )
        self.llm = 'gpt-4o-mini'

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
            verbose=True,
        )
