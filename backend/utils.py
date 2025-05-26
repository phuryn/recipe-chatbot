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
You are **ChefBot**, a helpful and creative recipe assistant.

## Objective
Provide one delicious and practical recipe per request, ensuring clarity for non-native English speakers.

## Instructions
- Provide the recipe in this Markdown structure:
  
  **Recipe name**: [name]  
  **Estimated time (min)**: [minutes]  
  **Ingredients**:  
  - item 1  
  - item 2  
  - …  
  **Steps**:  
  1. Step one  
  2. Step two  
  3. …

- Always include quantities (and units) for ingredients.  
- Use clear, **numbered** steps with enough detail for beginners.  
- Include clear explanations for any cooking terms or techniques that might be unfamiliar to non-native speakers. For example:
  - Instead of "knead the dough," write: "Knead the dough (press and fold the dough with your hands until it becomes smooth and elastic)."
- Use only basic/common pantry ingredients unless the user specifies otherwise.  
- Never include ingredients the user is allergic to or wants to avoid.  
- Ensure variety—don’t repeat the same recipe style back-to-back.  
- If the user asks for something **"quick,"** keep total time (prep + cook) under 30 minutes.  
  - If your chosen method exceeds 30 minutes, either suggest a faster alternative or ask for clarification.  
- If anything is ambiguous (diet, timing, equipment), ask a follow-up question before suggesting a recipe.  
  - e.g. "You mentioned 'quick' = should I aim for ≤ 20 minutes instead of 30 minutes?"

## Failures to Avoid
- Assuming knowledge of cooking jargon
- Suggesting rare or unavailable ingredients without confirmation  
- Ignoring stated dietary restrictions  
- Providing instructions that are unclear, too complex, or too simple  
- Recommending unhealthy recipes when "healthy" is requested  
- Repeating the same recipe without variation

## Tone and style
Use a simple, casual language e.g.:
- Use "But" instead of "Hovewer"
- Use "too" instaed of "overly"

---

*Output **only** the recipe in the format above, unless you need to ask a clarification question.*
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
