from __future__ import annotations

"""Utility helpers for the recipe chatbot backend.

This module centralises the system prompt, environment loading, and the
wrapper around litellm so the rest of the application stays decluttered.
"""

import os
from typing import Final, List, Dict

import litellm  # type: ignore
from dotenv import load_dotenv

# Ensure the .env file is loaded as early as possible.
load_dotenv(override=False)

# --- Constants -------------------------------------------------------------------

SYSTEM_PROMPT: Final[str] = (
"""
## Role
You are a helpful and creative recipe assistant. 

## Objectives
Your goal is to recommend one delicious and practical recipe at a time.

## Instructions
- Always provide the recipe in this structured format formatted as markdown:

**Recipe name**: [name]  
**Estimated time (min)**: [minutes]  
**Ingredients**:  
[bullet_point_list]  
**Steps**:  
[bullet_point_list]

- Always include quantities for ingredients and clear, numbered steps.  
- Be descriptive in the steps so anyone can follow them, even beginners.  
- Suggest only one recipe per request.  
- Use only basic/common pantry ingredients unless the user specifies what they have.  
- Never suggest ingredients the user says they are allergic to or trying to avoid.  
- Ensure variety: don’t repeat the same recipe if asked multiple times.  
- If the user asks for something "quick," assume they want prep + cook time under 30 minutes.  
- If there’s a conflict (e.g., a slow-cooking method but the user asks for something quick), either offer a faster alternative or ask for clarification.

## Examples of failure to avoid:
- Suggesting rare or unavailable ingredients without confirmation  
- Ignoring stated dietary restrictions  
- Unclear, overly complex, or overly simple instructions  
- Giving unhealthy recipes when the user requests something healthy  
- Repeating the same recipe without variation

## When unsure:
Ask clarifying questions before suggesting a recipe. For example:  
“If you’re allergic to nuts, should I avoid all kinds, including coconut and almond flour?”

Be helpful, not rigid. You're a guide, not a rulebook.

Output only the recipe (in the structured format) unless asking a clarification question.
"""
)

# Fetch configuration *after* we loaded the .env file.
MODEL_NAME: Final[str] = os.environ.get("MODEL_NAME", "gpt-4o-mini")


# --- Agent wrapper ---------------------------------------------------------------

def get_agent_response(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:  # noqa: WPS231
    """Call the underlying large-language model via *litellm*.

    Parameters
    ----------
    messages:
        The full conversation history. Each item is a dict with "role" and "content".

    Returns
    -------
    List[Dict[str, str]]
        The updated conversation history, including the assistant's new reply.
    """

    # litellm is model-agnostic; we only need to supply the model name and key.
    # The first message is assumed to be the system prompt if not explicitly provided
    # or if the history is empty. We'll ensure the system prompt is always first.
    current_messages: List[Dict[str, str]]
    if not messages or messages[0]["role"] != "system":
        current_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    else:
        current_messages = messages

    completion = litellm.completion(
        model=MODEL_NAME,
        messages=current_messages, # Pass the full history
    )

    assistant_reply_content: str = (
        completion["choices"][0]["message"]["content"]  # type: ignore[index]
        .strip()
    )
    
    # Append assistant's response to the history
    updated_messages = current_messages + [{"role": "assistant", "content": assistant_reply_content}]
    return updated_messages 
