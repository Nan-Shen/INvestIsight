import os
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from firecrawl import FirecrawlApp, JsonConfig
# FirecrawlScrapeWebsiteTool
from typing import Type
from dotenv import load_dotenv
from pathlib import Path
dotenv_path = Path('../keys/.env')
load_dotenv(dotenv_path=dotenv_path)

class ScrapeInput(BaseModel):
    """Input schema for ScrapeTool."""
    url: str = Field(description="The URL to be scraped.")

class ExtractSchema(BaseModel):
    company_mission: str
    supports_sso: bool
    is_open_source: bool
    is_in_yc: bool

class ScrapeTool(BaseTool):
    name: str = "Website scrape tool"
    description: str = "Scrapes a URL and get its content",
    args_schema: Type[BaseModel] = ScrapeInput
    extractionSchema: Type[BaseModel] = ExtractSchema
    api_key: str = os.getenv("FIRECRAWL_API_KEY")

    def _run(self, url: str) -> str:
        app = FirecrawlApp(api_key=self.api_key)
        json_config = JsonConfig(
            extractionSchema=ExtractSchema.model_json_schema(),
            mode="llm-extraction",
            pageOptions={"onlyMainContent": True}
        )
        response = app.scrape_url(url,
                                  formats=["json"],
                                    json_options=json_config)

        if response["metadata"]['statusCode'] != 200:
            return f"""Failed to fetch the data from
                       {response['metadata']['title']} at
                       {response['metadata']['url']}"""
        
        return response["markdown"]