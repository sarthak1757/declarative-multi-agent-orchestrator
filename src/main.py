import asyncio
from dotenv import load_dotenv
load_dotenv() # Load env vars before imports

import src.utils.logger # Initialize logging
import os
from src.utils.yaml_loader import load_yaml
from src.executor.executor import Executor
from src.agents.agent import Agent
from src.llm.openai_client import OpenAIClient
from src.llm.gemini_client import GeminiClient
from src.llm.mock_client import MockClient
from src.tools.registry import TOOL_REGISTRY

import sys

if os.getenv("MOCK_MODE") == "true":
    print("⚠️  RUNNING IN MOCK MODE")
    LLMS = {
        "openai": MockClient("mock-openai"),
        "gemini": MockClient("mock-gemini"),
        "claude": MockClient("mock-claude"),
    }
else:
    LLMS = {
        "openai": OpenAIClient(),
        "gemini": GeminiClient(),
        # "claude": ClaudeClient(), # TODO: Implement real Claude client
    }

async def run(yaml_file):
    config = load_yaml(yaml_file)

    agents = {}
    for a in config["agents"]:
        agents[a["id"]] = Agent(
            a["id"],
            a.get("role"),
            a.get("goal"),
            LLMS.get(a.get("model", "openai")),
            [TOOL_REGISTRY[t] for t in a.get("tools", [])]
        )

    executor = Executor(agents)
    wf = config["workflow"]

    if wf["type"] == "sequential":
        print(await executor.sequential(wf["steps"]))

    elif wf["type"] == "parallel":
        print(await executor.parallel(wf["branches"], wf["then"]))

    elif wf["type"] == "supervisor":
        print(await executor.supervisor(wf["root"], wf["sub_agents"]))

if __name__ == "__main__":
    yaml_file = sys.argv[1] if len(sys.argv) > 1 else "examples/sequential.yaml"
    asyncio.run(run(yaml_file))#