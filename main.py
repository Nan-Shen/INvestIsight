#!/usr/bin/env python
from random import randint
from pydantic import BaseModel
from crewai.flow import Flow, listen, start
from crews.invest_crew.invest_crew import InvestCrew
from crews.plan_crew.plan_crew import PlanCrew
import os
import asyncio
import click
import datetime

from dotenv import load_dotenv
from pathlib import Path
dotenv_path = Path('./keys/.env')
load_dotenv(dotenv_path=dotenv_path)

class Section(BaseModel):
    title: str = ""
    content: str = ""

class ReportState(BaseModel):
    query: str = "Current status and potential risk factors of Novo Nordisk"
    titles: list[str] = []
    points: list[list] = []
    sections: list[Section] = []
    n_sections: int = 0


class SectionFlow(Flow[ReportState]):

    @start()
    def generate_plan(self, **kwargs):
        print("Generating investigation plan")
        # generate outline on the "topic"
        outline = PlanCrew().crew().kickoff(inputs={"query": self.state.query})
        # collect total outline and main points from the crew output
        insight_outlines = outline.pydantic.sections
        self.state.n_sections = len(insight_outlines)
        self.state.titles = [section.heading for section in insight_outlines]
        self.state.points = [section.bullet_points for section in insight_outlines]
    
    @listen(generate_plan)
    async def generate_tasks(self):
        print("Investigating sections")
        tasks = []
        async def single_task(title: str, points: list):
            result = (
                InvestCrew()
                .crew()
                .kickoff(inputs={
                    "topic": title,
                    "points": ';'.join(points),
                    "query": self.state.query,
                    "sections": [section.title for section in self.state.sections]
                })
            )
            return result.pydantic
    
        # Create tasks for each section
        for i in range(self.state.n_sections):
            task = asyncio.create_task(single_task(self.state.titles[i], self.state.points[i]))
            tasks.append(task)
    
        # Wait for all tasks to be generated concurrently
        sections = await asyncio.gather(*tasks)
        print(f"Generated {len(sections)} sections")
        self.state.sections.extend(sections)

    @listen(generate_tasks)
    def publish_report(self):
        # Print a summary notification
        print(f"✅ Report Completed!\n")
    
        text = ""
        for section in self.state.sections:
            text += "## " + section.title + "\n\n" + section.content
            text += "\n\n ------ \n\n"
   
        filename = str(datetime.datetime.now())+".report.md"
        with open(filename, "w") as f:
            f.write(text)
            
        print(f"📄 Final report saved to {filename}")

def kickoff():
    flow = SectionFlow()
    res = flow.kickoff()
    print(res)

def plot():
    flow = SectionFlow()
    flow.plot()

if __name__ == "__main__":
    kickoff()
    
