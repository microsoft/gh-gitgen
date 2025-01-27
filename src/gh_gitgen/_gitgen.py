import argparse
import asyncio
import logging
import subprocess
import sys

import aiohttp
import pyperclip
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import TextMessage, ToolCallSummaryMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient

logger = logging.getLogger("gitgen")
logger.addHandler(logging.StreamHandler())


async def get_github_issue_content(owner: str, repo: str, issue_number: int) -> str:
    issue_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
    comments_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments"

    async with aiohttp.ClientSession() as session:
        async with session.get(issue_url) as issue_response:
            if issue_response.status == 200:
                issue = await issue_response.json()
                issue_content = issue.get("body", "No content")
                issue_user = issue.get("user", {}).get("login", "Unknown user")
            else:
                return f"Error fetching issue: {issue_response.status}"

        async with session.get(comments_url) as comments_response:
            if comments_response.status == 200:
                comments = await comments_response.json()
                comments_content = "\n\n".join(
                    [
                        f"{comment.get('user', {}).get('login', 'Unknown user')} (ID: {comment.get('user', {}).get('id', 'Unknown ID')}): {comment.get('body', 'No content')}"
                        for comment in comments
                    ]
                )
            else:
                return f"Error fetching comments: {comments_response.status}"

    return f"Issue Content by {issue_user}:\n{issue_content}\n\nComments:\n{comments_content}"


async def run(agent: AssistantAgent, task: str, log: bool=True) -> str:
    output_stream = agent.on_messages_stream(
        [TextMessage(content=task, source="user")],
        cancellation_token=CancellationToken(),
    )
    last_txt_message = ""

    async for message in output_stream:
        if isinstance(message, Response):
            if isinstance(message.chat_message, TextMessage):
                last_txt_message += message.chat_message.content
            elif isinstance(message.chat_message, ToolCallSummaryMessage):
                last_txt_message += message.chat_message.content
            else:
                raise ValueError(f"Unexpected message type: {message.chat_message}")
            if log:
                print(last_txt_message)
    return last_txt_message


async def get_user_confirmation(prompt: str) -> bool:
    user_input = await get_user_input(f"{prompt} (y to confirm, or provide feedback)")
    user_input = user_input.lower().strip()
    return user_input == "y"


async def get_user_input(prompt: str) -> str:
    return input(f"\n>> {prompt}: ").strip()


async def gitgen(owner: str, repo: str, command: str, number: int):
    print(f"Processing: {command} #{number} for {owner}/{repo}")

    agent = AssistantAgent(
        name="GitGenAgent",
        system_message="You are a helpful AI assistant whose purpose is to reply to GitHub issues and pull requests. Use the content in the thread to generate an auto reply that is technical and helpful to make progress on the issue/pr. Your response must be very concise and focus on precision. Just be direct and to the point.",
        model_client=OpenAIChatCompletionClient(model="gpt-4o"),
        tools=[get_github_issue_content],
    )
    task = f"Fetch comments for the {command} #{number} for the {owner}/{repo} repository"
    await run(agent, task, log=True)

    print("Thinking...")
    await run(
        agent,
        "Answer the following questions: 1) What facts are known based on the contents of this issue thread? 2) What is the main issue or problem that needs. 3) What type of a new response from the maintainers would help make progress on this issue? Be concise.",
        log=False,
    )

    print("\nSummary: ")
    await run(agent, "Summarize what is the status of this issue. Be concise.")

    print("\nSuggested response: ")
    suggested_response = await run(
        agent,
        "On behalf of the maintainers, generate a response to the issue/pr that is technical and helpful to make progress. Be concise.",
    )

    while True:
        user_feedback = await get_user_input("Provide feedback on the suggested response")
        if user_feedback.lower().strip() == "exit":
            print("Exiting...")
            break
        if user_feedback.lower().strip() == "y":
            print("Replying to the issue...")
            # Copy the suggested response to the clipboard
            pyperclip.copy(suggested_response)
            print("The suggested response has been copied to your clipboard.")
            break
        else:
            print("\nSuggested response:")
            suggested_response = await run(
                agent,
                f"Accommodate the following feedback: {user_feedback}. Then generate a response to the issue/pr that is technical and helpful to make progress. Be concise.",
            )


def main():
    parser = argparse.ArgumentParser(description="Process GitHub issues or pull requests.")
    parser.add_argument(
        "--repo",
        help="Repository info in the format 'owner/repo', "
        "if not provided, it will be detected based on the repo "
        "in the current directory.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("command", choices=["issue", "pr"], help="Command to execute (issue or pr)")
    parser.add_argument("number", type=int, help="Issue or PR number")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)

    if args.repo:
        owner, repo = args.repo.split("/")
    else:
        # Detect the owner and repo based on the current directory using subprocess.
        pipe = subprocess.run(
            [
                "gh",
                "repo",
                "view",
                "--json",
                "owner,name",
                "-q",
                '.owner.login + "/" + .name',
            ],
            check=True,
            capture_output=True,
        )
        owner, repo = pipe.stdout.decode().strip().split("/")

    command = args.command
    number = args.number

    if command == "issue":
        asyncio.run(gitgen(owner, repo, command, number))
    else:
        print(f"Command '{command}' is not implemented.")
        sys.exit(1)


if __name__ == "__main__":
    main()
