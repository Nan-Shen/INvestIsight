from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import SerperDevTool
from typing import List, Type
from pydantic import BaseModel, Field

from invest_agent.tools.custom_tool import ScrapeTool


class Outline(BaseModel):
    sections: List[InsightOutline] = Field(
        description="A list of outline sections defining the article structure."
    )
    
class InsightOutline(BaseModel):
    heading: str = Field( 
        description="The title of the outline section, serving as a main heading."
    )
    bullet_points: List[str] = Field(
        description="A list of key points summarizing this section."
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
class ValidateCrew:
    """Investigation Crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self):
        self.search_tool = SerperDevTool()
        self.srcape_tool = ScrapeTool()
        self.llm = LLM(
                    model="ollama/llama3.2:1b",
                    base_url="http://localhost:11434"
        )

    def publish_article(self, output):
        # Print a summary notification
        print(f"✅ Report Completed!\nTask: {output.description}")
    
        text = ""
        for section in output.pydantic.sections:
            text += "## " + section.title + "\n\n" + section.content
            text += "\n\n ------ \n\n"
            
        filename = "report.md"
        with open(filename, "w") as f:
            f.write(text)
            
        print(f"📄 Final report saved to {filename}")
                    
    @agent
    def validate_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["validate_agent"],
            llm=self.llm,
            allow_delegation=True,
            tools=[self.search_tool, self.srcape_tool],
        )
        
    @task
    def validate_task(self) -> Task:
        return Task(
            config=self.tasks_config["validate_task"],
            output_pydantic=Report,
            tools=[self.search_tool, self.srcape_tool],
            human_input=True,
            callback=self.publish_article
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Research Crew"""
        return Crew(
            agents=self.agents, 
            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True,
        )
