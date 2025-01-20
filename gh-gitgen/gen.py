import sys
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main(task):
    print(f"Processing task: {task}")
    agent = AssistantAgent("assistant", OpenAIChatCompletionClient(model="gpt-4o"))
    response = await agent.run(task=task)
    print(f"Agent response: {response}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: No task provided. Usage: gh gen <task>")
        sys.exit(1)

    task_description = " ".join(sys.argv[1:])
    asyncio.run(main(task_description))
