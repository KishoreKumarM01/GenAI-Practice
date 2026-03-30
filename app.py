import re
import os
import time
from aiohttp import web
from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings
from botbuilder.schema import Activity
from dotenv import load_dotenv

from rag_chain import load_chain
from summariser import generate_dev_notes, generate_consolidated_notes
from confluence_client import (
    get_page_by_url, get_page_by_title,
    get_page_title_from_url, extract_text
)
from graph_client import (
    get_conversation,
    extract_confluence_links,
    filter_relevant_links
)

load_dotenv()

# ─── Bot Setup ───────────────────────────────────────────────────────────────
SETTINGS = BotFrameworkAdapterSettings(
    os.getenv("BOT_APP_ID", ""),
    os.getenv("BOT_APP_PASSWORD", "")
)
ADAPTER = BotFrameworkAdapter(SETTINGS)

print("Loading RAG chain...")
_chain = load_chain()
print("RAG chain ready.")


# ─── TTL Dict ────────────────────────────────────────────────────────────────
class TTLDict:
    """
    Dict that auto-expires entries after ttl_seconds.
    Prevents stale pending state from accumulating in memory.
    Default: 5 minutes.
    """
    def __init__(self, ttl_seconds=300):
        self._data = {}
        self._ttl = ttl_seconds

    def set(self, key, value):
        self._data[key] = {"value": value, "expires": time.time() + self._ttl}

    def get(self, key):
        entry = self._data.get(key)
        if not entry:
            return None
        if time.time() > entry["expires"]:
            del self._data[key]
            return None
        return entry["value"]

    def pop(self, key):
        entry = self._data.pop(key, None)
        if not entry:
            return None
        if time.time() > entry["expires"]:
            return None
        return entry["value"]

    def __contains__(self, key):
        return self.get(key) is not None


# ─── Pending state ────────────────────────────────────────────────────────────
# Keyed by conv_id::user_id — isolates state per user per conversation
# TTL of 5 minutes — stale entries auto-expire
_pending_link_choice = TTLDict(ttl_seconds=300)
_pending_mode_choice = TTLDict(ttl_seconds=300)


def make_pending_key(turn_context) -> str:
    """
    Unique key per user per conversation.
    Prevents one user's pending state from affecting another's
    even when they are in the same channel or group chat.
    """
    user_id = (
        turn_context.activity.from_property.id
        if turn_context.activity.from_property
        else "unknown"
    )
    conv_id = turn_context.activity.conversation.id
    return f"{conv_id}::{user_id}"


# ─── Playground test data ─────────────────────────────────────────────────────
PLAYGROUND_CONVERSATION = """
Ravi: the claims API is failing on duplicate submissions
Srinivas: yeah I saw that — no idempotency check exists in the adjudication service
Ravi: should we use Redis for idempotency keys?
Anand: yes Redis makes sense, TTL 24 hours should be enough
Srinivas: ok I will implement it and raise a PR today
Ravi: needs to be done before sprint 24 ends
Anand: also someone needs to update the Confluence spec page after
Ravi: I will do that once Srinivas finishes the implementation
Srinivas: one blocker — Redis is not provisioned in prod yet
Anand: I will raise that with DevOps today
"""

PLAYGROUND_CONFLUENCE = """
Claims Adjudication Process
Claims are submitted by providers via EDI 837 transaction.
Steps:
1. Claim Submission - Provider submits claim
2. Initial Review - System validates member eligibility
3. Medical Review - Clinical team reviews procedure codes
4. Adjudication Decision - Claim approved, denied, or pended
5. Payment Processing - EOB generated and payment issued
6. Denial Management - Denied claims trigger denial reason codes
Key Rules:
- Claims must be submitted within 180 days of service date
- Duplicate claims are auto-rejected by the system
- High-cost claims above $50,000 require manual review
"""

URL_PATTERN = r'https://[a-zA-Z0-9-]+\.atlassian\.net/wiki/[^\s\)\]>"]+'

CONTEXT_LABELS = {
    "channel":    "Teams channel",
    "group_chat": "group chat",
    "playground": "playground (test mode)"
}


# ─── Command parser ───────────────────────────────────────────────────────────
def parse_dev_note_command(text):
    """
    Parse [DevNoteSummarise] command.

    Valid formats:
      [DevNoteSummarise]
      [DevNoteSummarise] live
      [DevNoteSummarise] rag
      [DevNoteSummarise] https://url live
      [DevNoteSummarise] https://url rag
      [DevNoteSummarise] https://url        <- asks live/rag

    Returns dict: { url: str|None, mode: str|None }
    Returns None if command token not found.
    """
    if not re.search(r'\[DevNoteSummarise\]', text, re.IGNORECASE):
        return None

    remainder = re.sub(r'\[DevNoteSummarise\]', '', text, flags=re.IGNORECASE).strip()

    url_match = re.search(URL_PATTERN, remainder)
    url = url_match.group(0) if url_match else None

    without_url = remainder.replace(url, "").strip() if url else remainder

    if re.search(r'\blive\b', without_url, re.IGNORECASE):
        mode = "live"
    elif re.search(r'\brag\b', without_url, re.IGNORECASE):
        mode = "rag"
    else:
        mode = None

    return {"url": url, "mode": mode}


# ─── Main message handler ─────────────────────────────────────────────────────
async def handle_turn(turn_context):
    
        # Ignore non-message activities silently
    if turn_context.activity.type != "message":
        # Optionally send a welcome message on install
        if turn_context.activity.type == "installationUpdate":
            await turn_context.send_activity(
                "Hi! I am your Confluence BA Assistant.\n"
                "Type [DevNoteSummarise] to generate dev notes, "
                "or ask me anything about your Confluence docs."
            )
        return
    
    text = (turn_context.activity.text or "").strip()
    pending_key = make_pending_key(turn_context)

    # ── Pending: live vs RAG choice (per user per conversation) ──────────────
    if pending_key in _pending_mode_choice:
        pending = _pending_mode_choice.pop(pending_key)
        await resolve_mode_choice(turn_context, text, pending)
        return

    # ── Pending: multi-link selection (per user per conversation) ─────────────
    if pending_key in _pending_link_choice:
        pending = _pending_link_choice.pop(pending_key)
        await resolve_link_choice(turn_context, text, pending)
        return

    # ── [DevNoteSummarise] command ────────────────────────────────────────────
    parsed = parse_dev_note_command(text)
    if parsed is not None:
        await handle_dev_note(turn_context, parsed)
        return

    # ── Greeting ──────────────────────────────────────────────────────────────
    if re.search(r"^(hello|hi|hey|greetings)$", text, re.IGNORECASE):
        await turn_context.send_activity(
            "Hi! I am your Confluence BA Assistant.\n\n"
            "Dev Notes Command:\n"
            "[DevNoteSummarise] - summarise conversation, auto-detect page (RAG)\n"
            "[DevNoteSummarise] live - auto-detect page, Live fetch\n"
            "[DevNoteSummarise] https://url live - specific page, Live\n"
            "[DevNoteSummarise] https://url rag - specific page, RAG\n"
            "[DevNoteSummarise] https://url - specific page, will ask live/rag\n\n"
            "Works in: Teams channels and group chats\n"
            "Each user's session is isolated — no cross-user interference.\n\n"
            "Or ask me anything about your Confluence docs."
        )
        return

    # ── Default: RAG Q&A ─────────────────────────────────────────────────────
    await turn_context.send_activity("Searching Confluence...")
    result = _chain.invoke(text)
    answer = result["result"]
    sources = "\n".join([
        f"- {doc.metadata['title']}: {doc.metadata['url']}"
        for doc in result["source_documents"]
    ])
    await turn_context.send_activity(f"{answer}\n\nSources:\n{sources}")


# ─── [DevNoteSummarise] handler ───────────────────────────────────────────────
async def handle_dev_note(turn_context, parsed):
    pending_key = make_pending_key(turn_context)

    # Step 1 — Fetch conversation
    try:
        conversation, context_type, _ = get_conversation(turn_context, limit=50)
    except Exception as e:
        await turn_context.send_activity(f"Failed to fetch messages: {str(e)}")
        return

    is_playground = context_type == "playground"
    context_label = CONTEXT_LABELS.get(context_type, context_type)

    if is_playground:
        await turn_context.send_activity(
            f"Playground mode - using test conversation.\n"
            f"In production this reads from your {context_label}."
        )
        conversation = PLAYGROUND_CONVERSATION
    else:
        await turn_context.send_activity(
            f"Fetching messages from {context_label}..."
        )

    if not conversation.strip():
        await turn_context.send_activity("No messages found to summarise.")
        return

    # Step 2 — Determine Confluence page
    explicit_url = parsed.get("url")
    mode = parsed.get("mode")

    if explicit_url:
        page_title = get_page_title_from_url(explicit_url)
        await turn_context.send_activity(f"Using page: {page_title}")

        if mode is None:
            # Ask user which mode — stored under their unique pending key
            _pending_mode_choice.set(pending_key, {
                "url": explicit_url,
                "page_title": page_title,
                "conversation": conversation,
                "is_playground": is_playground
            })
            await turn_context.send_activity(
                f"Page: {page_title}\n\n"
                f"Which data source?\n\n"
                f"1. Live - fetch directly from Confluence (always current)\n"
                f"2. RAG - use indexed knowledge base (fast, up to 24hrs old)\n\n"
                f"Reply 1 for Live or 2 for RAG.\n"
                f"(This selection will expire in 5 minutes if not answered.)"
            )
            return

        await fetch_and_generate(
            turn_context, conversation, explicit_url, page_title, mode, is_playground
        )
        return

    # No URL in command — scan conversation for links
    links = extract_confluence_links(conversation)
    links = filter_relevant_links(links, conversation)

    if len(links) == 0:
        await turn_context.send_activity(
            "No Confluence links found in conversation.\n"
            "Auto-detecting most relevant page from knowledge base...\n"
            "Note: Recently created or updated pages may not be indexed yet."
        )
        page_meta = rag_auto_detect(conversation)

        if page_meta:
            await turn_context.send_activity(
                f"Auto-detected: {page_meta['title']}\n"
                f"To use a different page: [DevNoteSummarise] https://url live/rag"
            )
            effective_mode = mode if mode else "live"
            await fetch_and_generate(
                turn_context, conversation,
                page_meta["url"], page_meta["title"],
                effective_mode, is_playground
            )
        else:
            await turn_context.send_activity(
                "No relevant page found — generating from conversation only..."
            )
            notes = generate_dev_notes(conversation)
            await turn_context.send_activity(notes)

    elif len(links) == 1:
        page_title = get_page_title_from_url(links[0])
        await turn_context.send_activity(f"Found page: {page_title}")
        effective_mode = mode if mode else "live"
        await fetch_and_generate(
            turn_context, conversation, links[0],
            page_title, effective_mode, is_playground
        )

    elif len(links) <= 4:
        options = "\n".join([
            f"{i + 1}. {get_page_title_from_url(link)}"
            for i, link in enumerate(links)
        ])
        # Store under unique pending key — isolated per user
        _pending_link_choice.set(pending_key, {
            "links": links,
            "mode": mode,
            "conversation": conversation,
            "is_playground": is_playground
        })
        await turn_context.send_activity(
            f"Found {len(links)} Confluence pages in this conversation:\n\n"
            f"{options}\n\n"
            f"Reply with the number to use, or 'all' to combine them.\n"
            f"(This selection will expire in 5 minutes if not answered.)"
        )

    else:
        await turn_context.send_activity(
            f"Found {len(links)} Confluence links - too many to auto-select.\n"
            f"Specify one: [DevNoteSummarise] https://url live/rag"
        )


# ─── Fetch page and generate notes ───────────────────────────────────────────
async def fetch_and_generate(
    turn_context, conversation, url, page_title, mode, is_playground
):
    if mode == "live":
        await turn_context.send_activity(f"Fetching live: {page_title}...")
        page = get_page_by_url(url)
        if not page:
            await turn_context.send_activity(
                "Could not fetch that page. Try RAG mode or check the URL."
            )
            return
        content = extract_text(page)
        actual_title = page.get("title", page_title)
        source_label = "Source: Live Confluence fetch"
    else:
        await turn_context.send_activity(
            f"Searching knowledge base: {page_title}..."
        )
        page = get_page_by_title(page_title)
        content = extract_text(page) if page else ""
        actual_title = page.get("title", page_title) if page else page_title
        source_label = "Source: Indexed knowledge base (may be up to 24hrs old)"

    if is_playground and not content.strip():
        content = PLAYGROUND_CONFLUENCE
        actual_title = "Claims Adjudication Process (test)"

    if not content.strip():
        await turn_context.send_activity(
            f"Could not retrieve content for {page_title}.\n"
            f"Generating from conversation only..."
        )
        notes = generate_dev_notes(conversation)
        await turn_context.send_activity(notes)
        return

    await turn_context.send_activity("Generating consolidated dev notes...")
    notes = generate_consolidated_notes(
        conversation=conversation,
        confluence_content=content,
        page_title=actual_title
    )
    await turn_context.send_activity(f"{notes}\n\n{source_label}")


# ─── RAG auto-detect ─────────────────────────────────────────────────────────
def rag_auto_detect(conversation):
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_openai import AzureOpenAIEmbeddings
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("OPENAI_API_VERSION")
        )
        vectorstore = Chroma(
            persist_directory="./confluence_index",
            embedding_function=embeddings
        )
        results = vectorstore.similarity_search(conversation[:2000], k=1)
        if results:
            return results[0].metadata
    except Exception:
        pass
    return None


# ─── Pending: live vs RAG choice ─────────────────────────────────────────────
async def resolve_mode_choice(turn_context, selection, pending):
    selection = selection.strip()
    if selection == "1" or "live" in selection.lower():
        mode = "live"
    elif selection == "2" or "rag" in selection.lower():
        mode = "rag"
    else:
        await turn_context.send_activity(
            "Please reply 1 for Live or 2 for RAG."
        )
        # Put back with fresh TTL
        _pending_mode_choice.set(make_pending_key(turn_context), pending)
        return

    await fetch_and_generate(
        turn_context,
        pending["conversation"],
        pending["url"],
        pending["page_title"],
        mode,
        pending["is_playground"]
    )


# ─── Pending: multi-link selection ───────────────────────────────────────────
async def resolve_link_choice(turn_context, selection, pending):
    links = pending["links"]
    mode = pending["mode"] if pending["mode"] else "live"
    conversation = pending["conversation"]
    is_playground = pending["is_playground"]

    if selection.strip().lower() == "all":
        await turn_context.send_activity("Combining all pages...")
        combined_content = ""
        for link in links:
            page = get_page_by_url(link)
            if page:
                combined_content += f"\n\n--- {page.get('title', '')} ---\n"
                combined_content += extract_text(page)
        if is_playground and not combined_content.strip():
            combined_content = PLAYGROUND_CONFLUENCE
        await turn_context.send_activity("Generating consolidated dev notes...")
        notes = generate_consolidated_notes(
            conversation=conversation,
            confluence_content=combined_content,
            page_title="Multiple Pages"
        )
        await turn_context.send_activity(notes)
        return

    try:
        idx = int(selection.strip()) - 1
        if 0 <= idx < len(links):
            link = links[idx]
            page_title = get_page_title_from_url(link)
            await turn_context.send_activity(f"Using: {page_title}")
            await fetch_and_generate(
                turn_context, conversation, link, page_title, mode, is_playground
            )
        else:
            await turn_context.send_activity(
                "Invalid number. Please try the command again."
            )
    except ValueError:
        await turn_context.send_activity(
            "Please reply with a number or 'all'."
        )


# ─── Web server ───────────────────────────────────────────────────────────────
async def messages(req: web.Request) -> web.Response:
    body = await req.json()
    activity = Activity().deserialize(body)
    auth_header = req.headers.get("Authorization", "")
    await ADAPTER.process_activity(activity, auth_header, handle_turn)
    return web.Response(status=200)


async def health(req: web.Request) -> web.Response:
    return web.Response(status=200)


app = web.Application()
app.router.add_post("/api/messages", messages)
app.router.add_get("/api/messages", health)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 3978))
    print(f"Bot running on port {port}")
    web.run_app(app, port=port)