# INvestInsight

Welcome to the INvestInsight project, powered by [crewAI](https://crewai.com). This agent is designed to help you get insights for business and finacial status of any publicly traded companies with ease using a multi-agent AI system. 

## Running the Project
Put your query as the query value of ReportState in main.py. And run main.py.

## How it works

The INvestInsight Crew is composed of multiple AI agents, each with unique roles, goals, and tools. These agents collaborate on a series of tasks, leveraging their collective skills to achieve complex objectives. Here's how it works:

    🧭 Planner Agent – Breaks down complex objectives into actionable steps.

    🔄 ReAct Agents – Each step is handled by a ReAct agent equipped with web search and scraping tools.

    ✅ Evaluator Agent – Verifies and ensures the accuracy and coherence of the outputs.

    👤 Human-in-the-Loop – Users can interact during the planning and evaluation phases to maintain relevance and domain alignment.
    
