import sys
import asyncio
import argparse
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

async def assistant_run_stream(agent: AssistantAgent, task: str, log=True) -> None:
    output_stream = agent.on_messages_stream(
                [TextMessage(content=task, source="user")],
                cancellation_token=CancellationToken(),
            )
    async for message in output_stream:
        if log:
            print(message.chat_message.content)

async def main(owner: str, repo: str, command: str, number: int):
    
    print(f"Processing: {command} #{number} for {owner}/{repo}")
    
    agent = AssistantAgent(
        name="GitGenAgent",
        system_message="You are a helpful AI assistant whose purpose is to reply to GitHub issues and pull requests. Use the content in the thread to generate an auto reply that is technical and helpful to make progress on the issue/pr. Your response must be very concise and focus on precision. Just be direct and to the point.",
        model_client=OpenAIChatCompletionClient(model="gpt-4o"),
        tools=[get_github_issue_content]
    )
    task = f"Fetch comments for the {command} #{number} for the {owner}/{repo} repository"
    await assistant_run_stream(agent, task, log=False)

    print("Thinking...")
    await assistant_run_stream(agent, "Answer the following questions: 1) What facts are known based on the contents of this issue thread? 2) What is the main issue or problem that needs. 3) What type of a new response from the maintainers would help make progress on this issue? Be concise.", log=False)

    print("\nSummary: ")
    await assistant_run_stream(agent, "Summarize what is the status of this issue. Be concise.")

    print("\nSuggested response: ")
    await assistant_run_stream(agent, "On behalf of the maintainers, generate a response to the issue/pr that is technical and helpful to make progress. Be concise.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process GitHub issues or pull requests.")
    parser.add_argument("repo_info", help="Repository information in the format 'owner/repo'")
    parser.add_argument("command", choices=["issue", "pr"], help="Command to execute (issue or pr)")
    parser.add_argument("number", type=int, help="Issue or PR number")

    args = parser.parse_args()

    owner, repo = args.repo_info.split('/')
    command = args.command
    number = args.number

    if command == "issue":
        asyncio.run(main(owner, repo, command, number))
    else:
        print(f"Command '{command}' is not implemented.")
        sys.exit(1)
