from gpt_researcher import GPTResearcher
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_aws import BedrockEmbeddings

import asyncio
from enum import Enum
import json
import tempfile
import os


class Provider(Enum):
    OPENAI = {
        "FAST_LLM": "openai:gpt-4.1-mini-2025-04-14",
        "SMART_LLM": "openai:gpt-4.1-mini-2025-04-14",
        "STRATEGIC_LLM": "openai:gpt-4.1-mini-2025-04-14"
    }
    ANTHROPIC = {
        "FAST_LLM": "anthropic:claude-3-5-sonnet-latest",
        "SMART_LLM": "anthropic:claude-3-5-sonnet-latest",
        "STRATEGIC_LLM": "anthropic:claude-3-5-sonnet-latest"
    }
    GEMINI = {
        "FAST_LLM": "google_genai:gemini-2.0-flash-001",
        "SMART_LLM": "google_genai:gemini-2.0-flash-001",
        "STRATEGIC_LLM": "google_genai:gemini-2.0-flash-001"
    }

class ReportSource(Enum):
    Web = "web"
    Internal = "langchain_vectorstore"
    Hybrid = "hybrid"

def get_researcher(
        query: str,
        report_source: ReportSource,
        provider: Provider
    ) -> GPTResearcher:

    config_dict = {
        "RETRIEVER": "tavily",
        "EMBEDDING": "bedrock:amazon.titan-embed-text-v2:0"
    }
    config_dict.update(provider.value)
    config_dict["REPORT_SOURCE"] = report_source.value

    temp_config_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
    try:
        json.dump(config_dict, temp_config_file)
        temp_config_path = temp_config_file.name
    finally:
        temp_config_file.close()

    vector_store = None
    if report_source != ReportSource.Web:
        client = QdrantClient(
            url="https://<domain>:<port>",
            api_key="<your-api-key>",
        )
        vector_store = QdrantVectorStore(
            client=client,
            collection_name="<collection_name>",
            content_payload_key="content",
            embedding=BedrockEmbeddings(region_name="<region>", model_id="amazon.titan-embed-text-v2:0"),
        )

    researcher = GPTResearcher(
        query=query,
        config_path=temp_config_path,
        vector_store=vector_store)
    
    # print(json.dumps(researcher.cfg.__dict__, indent=2))
    
    os.remove(temp_config_path)

    return researcher


async def get_report(query: str, in_json: dict):
    provider_name = in_json.get("model_provider", "openai").upper()
    try:
        provider = Provider[provider_name]
    except KeyError:
        provider = Provider.OPENAI
    
    report_source_name = in_json.get("report_source", "web").lower()
    if report_source_name == "internal":
        report_source = ReportSource.Internal
    elif report_source_name == "hybrid":
        report_source = ReportSource.Hybrid
    else:
        report_source = ReportSource.Web
    
    researcher = get_researcher(query, report_source, provider)
        
    await researcher.conduct_research()
    report = await researcher.write_report()

    print(report)

if __name__ == "__main__":
    query = "Should I invest in Nvidia?"
    raw_json = """
{
  "model_provider": "gemini",
  "report_source": "web"
}
"""
    in_json = json.loads(raw_json)
    
    asyncio.run(get_report(query, in_json))
    # rs = get_researcher(query=query, report_source=ReportSource.Internal, provider=Provider.GEMINI)