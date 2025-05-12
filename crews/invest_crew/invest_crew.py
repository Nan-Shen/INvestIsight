from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import SerperDevTool, FirecrawlScrapeWebsiteTool
from typing import List, Type
from pydantic import BaseModel, Field

class InsightOutline(BaseModel):
    title: str = Field( 
        description="The title of the outline section, serving as a main heading."
    )
    bullet_points: List[str] = Field(
        description="A list of key points summarizing this section."
    )
    
class Outline(BaseModel):
    titles: List[InsightOutline] = Field(
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
class InvestCrew:
    """Investigation Crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self):
        self.search_tool = SerperDevTool()
        self.srcape_tool = FirecrawlScrapeWebsiteTool()
        # self.llm = LLM(
        #             model="ollama/llama3.2:1b",
        #             base_url="http://localhost:11434"
        # )
        self.llm = 'gpt-4o-mini'

    @agent
    def topic_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["topic_researcher"],
            tools=[self.search_tool, self.srcape_tool],
            llm=self.llm,
            multimodal=True
        )
        
    @task
    def research_topic(self) -> Task:
        return Task(
            config=self.tasks_config["research_topic"],
            tools=[self.search_tool, self.srcape_tool],
            async_execution=False,
        )
        
    @agent
    def topic_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["topic_writer"],
            llm=self.llm,
            multimodal=True
        )
        
    @task
    def write_topic(self) -> Task:
        return Task(
            config=self.tasks_config["write_topic"],
            output_pydantic=Section,
            async_execution=False,
            context=[self.research_topic()] #fixed AttributeError: 'function' object has no attribute 'get'
        )

    @agent
    def validate_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["validate_agent"],
            llm=self.llm,
            allow_delegation=False,
            tools=[self.search_tool, self.srcape_tool],
            multimodal=True
        )
        
    @task
    def validate_task(self) -> Task:
        return Task(
            config=self.tasks_config["validate_task"],
            output_pydantic=Section,
            tools=[self.search_tool, self.srcape_tool],
            human_input=True,
            async_execution=False,
            context=[self.research_topic()]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

