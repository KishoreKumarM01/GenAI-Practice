from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()


def get_llm():
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        temperature=0
    )


def generate_dev_notes(conversation: str) -> str:
    """
    Generate structured notes from conversation only.
    Used when no Confluence page is available or provided.
    """
    llm = get_llm()

    prompt = PromptTemplate(
        input_variables=["conversation"],
        template="""You are a senior technical note-taker supporting Technology Architects.
Your notes will be used as official records of architectural decisions.
Accuracy is critical — these notes may be referenced months later.

STRICT RULES:
- Only extract information explicitly stated in the conversation
- If a name is not mentioned for an action item — write TBD, never guess
- If something is unclear or incomplete — flag it explicitly, do not interpret
- Never invent, assume, or paraphrase beyond what was said
- If two people said conflicting things — capture BOTH, do not pick one
- Incomplete discussions must be marked as PENDING, not resolved

CONVERSATION:
{conversation}

Respond in this EXACT format. Do not add or remove sections.

## Summary
[2-3 sentences only: what problem was discussed, what was decided, what is pending]
[If nothing was decided write: "Discussion in progress — no final decisions made"]

## Decisions Made
[Only include decisions explicitly confirmed in conversation]
[If none: "No decisions confirmed in this discussion"]
- [decision] — Confirmed by: [name if mentioned]

## Action Items
[Only include actions explicitly assigned or volunteered in conversation]
[If none: "No action items assigned"]
- [ ] [action] — Owner: [name or TBD] — Due: [date/sprint if mentioned or TBD]

## Open Questions / Blockers
[Questions raised but NOT answered in this conversation]
[Blockers explicitly mentioned]
[If none: "None identified"]
- [question or blocker] — Raised by: [name if mentioned]

## Technical Notes
[Specific technical details, numbers, system names, approaches mentioned]
[If none: "None discussed"]
- [detail]

## Flagged Items
[Anything incomplete, contradictory, or needing follow-up from THIS conversation]
[If none: "None"]
- [flag]

Never invent information.
Never resolve ambiguity by guessing.
Never omit a conflict — always surface it."""
    )

    chain = prompt | llm
    result = chain.invoke({"conversation": conversation})
    return result.content


def generate_consolidated_notes(
    conversation: str,
    confluence_content: str,
    page_title: str
) -> str:
    """
    Generate a final authoritative document by merging:
    - Existing Confluence page (official spec)
    - Latest Teams conversation (latest decisions)

    Conversation always takes precedence over Confluence.
    Contradictions must be explicitly called out — never silently merged.
    This document is used by Technology Architects as the source of truth.
    """
    llm = get_llm()
    today = datetime.now().strftime("%Y-%m-%d")

    prompt = PromptTemplate(
        input_variables=["confluence_content", "conversation", "page_title", "today"],
        template="""You are a senior technical documentation specialist supporting Technology Architects.

Technology Architects are pulled into multiple group chats daily.
Each chat produces decisions that may support, contradict, or completely replace existing documentation.
Your job is to produce the FINAL AUTHORITATIVE document that captures what was decided and why.
This document will be the official record — it must be precise, complete, and honest.

YOU HAVE TWO INPUTS:

INPUT 1 — EXISTING CONFLUENCE PAGE (Official Spec): "{page_title}"
{confluence_content}

INPUT 2 — LATEST TEAMS CONVERSATION (Most Recent Decisions):
{conversation}

PRECEDENCE RULES — follow these strictly:
1. Conversation is ALWAYS the latest source of truth
2. Where conversation EXPLICITLY contradicts Confluence — conversation wins, flag it clearly
3. Where conversation ADDS new information — include it as new
4. Where Confluence has content NOT touched by conversation — preserve it as still valid
5. Where conversation marks something as deprecated, wrong, or outdated — flag it as such
6. Where conversation is AMBIGUOUS about something in Confluence — do NOT assume, flag it

STRICT RULES — never violate these:
- NEVER silently merge contradictions — every conflict must be explicitly listed
- NEVER keep outdated content without marking it as outdated
- NEVER invent information not present in either source
- NEVER resolve ambiguity by guessing — flag it instead
- NEVER omit a conflict even if it seems minor — all conflicts must surface
- If a name is not mentioned for an action — write TBD
- If a date or sprint is not mentioned — write TBD
- If two people in the conversation disagreed — capture BOTH positions and mark as unresolved

Respond in this EXACT format. Do not add, remove, or rename any section.
Every section must be present even if the content is "None identified."

---

## Summary
[3 sentences maximum]
[Sentence 1: What this document covers and its current version]
[Sentence 2: What the latest conversation discussed or decided]
[Sentence 3: What changed from the previous spec, or "No changes from previous spec" if nothing changed]

---

## Current State (as of {today})
[The definitive authoritative position after applying conversation decisions]
[This section represents what is TRUE right now — not what was true before]
[Write in present tense — "The system currently does X" not "It was decided to do X"]
[If conversation did not change the current state — copy the relevant Confluence content here]

---

## Decisions Made
[Mark each decision clearly with its source and status]
[Format exactly as shown below — do not deviate]

- [NEW] [decision] — From: conversation — Confirmed by: [name or unknown]
- [MODIFIED] [old decision updated by conversation] — Was: [what Confluence said] — Now: [what conversation decided]
- [UNCHANGED] [decision preserved from Confluence not touched by conversation]
- [DEPRECATED] [decision or approach explicitly abandoned in conversation] — Reason: [why if stated]

[If no decisions at all: "No decisions confirmed — discussion ongoing"]

---

## Contradictions Resolved
[MOST CRITICAL SECTION — this is what makes this document valuable]
[List every single point where conversation contradicts Confluence]
[Even partial or indirect contradictions must be listed]
[Do not skip contradictions because they seem minor]

For each contradiction use this exact format:
CONFLICT: Confluence said "[copy exact relevant text from Confluence]"
CONVERSATION DECIDED: "[copy exact relevant text from conversation]"
RESOLUTION: Conversation takes precedence — [one sentence explanation of new position]
ACTION REQUIRED: [what needs to be updated, where, by whom if known]

[If absolutely no contradictions exist write exactly:]
"No contradictions identified between existing spec and latest discussion."

---

## Unresolved Items
[Items discussed but NOT fully resolved in this conversation]
[Disagreements between participants that were not settled]
[Questions raised but not answered]
[If none: "None identified"]

- [item] — Status: [why unresolved] — Next step: [if mentioned]

---

## Action Items
[Only include actions explicitly stated or volunteered in conversation]
[Do not create action items that were not discussed]
[If none: "No action items identified in this discussion"]

- [ ] [action] — Owner: [name or TBD] — Sprint/Due: [if mentioned or TBD] — Priority: [High/Medium/Low if implied]

---

## Open Questions / Blockers
[Questions raised but not answered in either source]
[Blockers explicitly mentioned]
[If none: "None identified"]

- [question or blocker] — Raised by: [name if mentioned] — Blocking: [what it blocks if clear]

---

## Technical Notes
[Specific technical details from BOTH sources]
[Conversation details take priority where they conflict with Confluence]
[Include: system names, version numbers, API names, thresholds, timeouts, configurations]
[If none: "None discussed"]

- [detail] — Source: [conversation/confluence]

---

## Preserved from Original Spec
[Content from Confluence page that was NOT discussed in conversation and remains valid]
[Do not include anything that was contradicted, deprecated, or modified above]
[If everything was covered in conversation: "All spec content addressed in latest discussion"]
[If nothing preserved: "No content preserved — full spec superseded by conversation"]

---

## Change Log
- {today} — Document updated based on Teams group chat discussion
  Changes made:
  - [bullet: what changed]
  - [bullet: what was added]
  - [bullet: what was deprecated]
  Previous spec said: [1-2 sentences summarising the old position before this update]
  Reason for change: [why the change was made, based on conversation context]
  Participants: [names mentioned in conversation if available]

---

FINAL REMINDER:
This document will be read by senior architects weeks or months from now.
They need to trust every word in it.
If you are unsure about anything — flag it. Do not guess.
A flagged uncertainty is far more valuable than a confident hallucination."""
    )

    chain = prompt | llm
    result = chain.invoke({
        "confluence_content": confluence_content,
        "conversation": conversation,
        "page_title": page_title,
        "today": today
    })
    return result.content