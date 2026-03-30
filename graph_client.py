import requests
import os
import re
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

SKIP_PATTERNS = ["devnotesummarise", "summarise", "summarize", "<at>"]


def get_graph_token() -> str:
    res = requests.post(
        f"https://login.microsoftonline.com/{os.getenv('TENANT_ID')}/oauth2/v2.0/token",
        data={
            "grant_type": "client_credentials",
            "client_id": os.getenv("AZURE_CLIENT_ID"),
            "client_secret": os.getenv("AZURE_CLIENT_SECRET"),
            "scope": "https://graph.microsoft.com/.default"
        }
    )
    res.raise_for_status()
    return res.json()["access_token"]


def _parse_messages(raw_messages: list) -> str:
    """Shared message parser — strips HTML, filters bot commands."""
    conversation = []
    for msg in reversed(raw_messages):
        sender = msg.get("from", {}).get("user", {}).get("displayName", "Unknown")
        body = msg.get("body", {}).get("content", "")
        text = BeautifulSoup(body, "html.parser").get_text().strip()
        if text and not any(p in text.lower() for p in SKIP_PATTERNS):
            conversation.append(f"{sender}: {text}")
    return "\n".join(conversation)


def get_channel_messages(team_id: str, channel_id: str, limit: int = 50) -> str:
    """Fetch messages from a Teams channel."""
    token = get_graph_token()
    res = requests.get(
        f"https://graph.microsoft.com/v1.0/teams/{team_id}/channels/{channel_id}/messages",
        headers={"Authorization": f"Bearer {token}"},
        params={"$top": min(limit, 50)}
    )
    res.raise_for_status()
    return _parse_messages(res.json().get("value", []))


def get_chat_messages(chat_id: str, limit: int = 50) -> str:
    """
    Fetch messages from a Teams group chat or 1:1 chat.
    Requires Chat.Read application permission.
    chat_id comes from activity.conversation.id e.g. 19:abc123@thread.v2
    """
    token = get_graph_token()
    res = requests.get(
        f"https://graph.microsoft.com/v1.0/chats/{chat_id}/messages",
        headers={"Authorization": f"Bearer {token}"},
        params={"$top": min(limit, 50)}
    )
    res.raise_for_status()
    return _parse_messages(res.json().get("value", []))


def detect_context(turn_context) -> str:
    """
    Determine the Teams context type from the activity.

    Returns:
      "channel"    - Teams channel message (has team_id + channel_id)
      "group_chat" - Group chat or 1:1 chat (has chat_id with @thread)
      "playground" - Local dev/test, no real Teams context
    """
    channel_data = turn_context.activity.channel_data or {}
    team_id = channel_data.get("team", {}).get("id") if channel_data else None
    channel_id = channel_data.get("channel", {}).get("id") if channel_data else None
    chat_id = turn_context.activity.conversation.id or ""

    if team_id and channel_id:
        return "channel"
    elif chat_id and "@thread" in chat_id:
        return "group_chat"
    else:
        return "playground"


def get_conversation(turn_context, limit: int = 50) -> tuple:
    """
    Fetch conversation based on context type.
    Returns (conversation_text, context_type, ids_dict)
    """
    context_type = detect_context(turn_context)
    channel_data = turn_context.activity.channel_data or {}

    if context_type == "channel":
        team_id = channel_data.get("team", {}).get("id")
        channel_id = channel_data.get("channel", {}).get("id")
        conversation = get_channel_messages(team_id, channel_id, limit)
        return conversation, context_type, {"team_id": team_id, "channel_id": channel_id}

    elif context_type == "group_chat":
        chat_id = turn_context.activity.conversation.id
        conversation = get_chat_messages(chat_id, limit)
        return conversation, context_type, {"chat_id": chat_id}

    else:
        return "", "playground", {}


def extract_confluence_links(conversation: str) -> list:
    """Extract all Confluence page links from a conversation."""
    pattern = r'https://[a-zA-Z0-9-]+\.atlassian\.net/wiki/[^\s\)\]>"]+'
    return list(set(re.findall(pattern, conversation)))


def filter_relevant_links(links: list, conversation: str) -> list:
    """Filter out links that conversation says to ignore."""
    ignore_signals = [
        "do not use", "old", "deprecated",
        "ignore", "wrong", "outdated", "dont use"
    ]
    relevant = []
    for link in links:
        idx = conversation.find(link)
        if idx == -1:
            relevant.append(link)
            continue
        surrounding = conversation[max(0, idx - 100):idx + 100].lower()
        if not any(signal in surrounding for signal in ignore_signals):
            relevant.append(link)
    return relevant