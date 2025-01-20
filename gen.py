import sys
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import aiohttp
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

# Define a tool to fetch the content of a GitHub issue and its comments
async def get_github_issue_content(owner: str, repo: str, issue_number: int) -> str:
    issue_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
    comments_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(issue_url) as issue_response:
            if issue_response.status == 200:
                issue = await issue_response.json()
                issue_content = issue.get("body", "No content")
            else:
                return f"Error fetching issue: {issue_response.status}"
        
        async with session.get(comments_url) as comments_response:
            if comments_response.status == 200:
                comments = await comments_response.json()
                comments_content = "\n\n".join([comment.get("body", "No content") for comment in comments])
            else:
                return f"Error fetching comments: {comments_response.status}"
    
    return f"Issue Content:\n{issue_content}\n\nComments:\n{comments_content}"

async def assistant_run_stream(agent: AssistantAgent, task: str) -> None:
    # Option 2: use Console to print all messages as they appear.
    await Console(
        agent.on_messages_stream(
            [TextMessage(content=task, source="user")],
            cancellation_token=CancellationToken(),
        )
    )

async def main(owner: str, repo: str, command: str, number: int):
    print(f"Processing task: {command} #{number} for repo {owner}/{repo}")
    agent = AssistantAgent(
        name="assistant",
        model_client=OpenAIChatCompletionClient(model="gpt-4o"),
        tools=[get_github_issue_content]
    )
    task = f"Fetch comments for the {command} #{number} for the {owner}/{repo} repository"
    await assistant_run_stream(agent, task)

    await assistant_run_stream(agent, "You are the maintainer of the repository. Draft a response to the issue.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: gen.py <owner/repo> <issue|pr> <number>")
        sys.exit(1)

    repo_info = sys.argv[1]
    command = sys.argv[2]
    number = int(sys.argv[3])  # Convert number to integer

    owner, repo = repo_info.split('/')

    if command == "issue":
        asyncio.run(main(owner, repo, command, number))
    else:
        print(f"Command '{command}' is not implemented.")
        sys.exit(1)
