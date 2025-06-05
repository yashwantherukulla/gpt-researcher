from gpt_researcher import GPTResearcher
import asyncio
import json

async def get_report(query: str, config_dict: dict):
    researcher = GPTResearcher(query, config_dict=config_dict)
    print(json.dumps(researcher.cfg.__dict__, indent=2))
    
    # await researcher.conduct_research()
    # report = await researcher.write_report()
    
    # research_context = researcher.get_research_context()
    # research_costs = researcher.get_costs()
    # research_images = researcher.get_research_images()
    # research_sources = researcher.get_research_sources()
    
    # return report, research_context, research_costs, research_images, research_sources

if __name__ == "__main__":
    query = "Should I invest in Nvidia?"
    config_dict = {
        "RETRIEVER": "tavily",
        "EMBEDDING": "google_genai:models/text-embedding-004",
        "FAST_LLM": "google_genai:gemini-2.0-flash-001",
        "SMART_LLM": "google_genai:gemini-2.0-flash-001",
        "STRATEGIC_LLM": "google_genai:gemini-2.0-flash-001",
        "REPORT_SOURCE": "web"
    }
    asyncio.run(get_report(query, config_dict))
    # report, context, costs, images, sources = asyncio.run(get_report(query, config_dict))
    
    # print("Report:")
    # print(report)
    # print("\nResearch Costs:")
    # print(costs)
    # print("\nResearch Images:")
    # print(images)
    # print("\nResearch Sources:")
    # print(sources)