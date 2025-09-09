import os
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx

DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
DEFAULT_LLM_MODEL = os.getenv("DEEPINFRA_LLM_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct")
DEFAULT_IMAGE_MODEL = os.getenv("DEEPINFRA_IMAGE_MODEL", "black-forest-labs/FLUX.1-dev")

OPENAI_COMPAT_BASE = "https://api.deepinfra.com/v1/openai"

app = FastAPI(title="DeepInfra Storyteller API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StoryRequest(BaseModel):
    genre: str = Field(example="fantasy")
    character_input: Optional[str] = Field(
        default=None,
        description="Either a comma-separated list of names OR a number like '3'",
    )
    paragraphs: int = Field(ge=1, le=12, example=5)
    audience: Optional[str] = Field(default="all ages", example="young adult")
    tone: Optional[str] = Field(default="whimsical", example="dark and suspenseful")
    include_images: bool = Field(default=True)
    include_preface: bool = Field(default=True)


async def call_deepinfra_chat(messages: list, model: str, response_json: bool = True) -> str:
    if not DEEPINFRA_API_KEY:
        raise HTTPException(status_code=500, detail="Missing DEEPINFRA_API_KEY env var")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.9,
    }
    if response_json:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{OPENAI_COMPAT_BASE}/chat/completions", headers=headers, json=payload)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        data = r.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Bad LLM response format: {e}")


async def call_deepinfra_images(prompt: str, model: str) -> Optional[str]:
    if not DEEPINFRA_API_KEY:
        raise HTTPException(status_code=500, detail="Missing DEEPINFRA_API_KEY env var")

    headers = {
        "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "prompt": prompt,
    }
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(f"{OPENAI_COMPAT_BASE}/images/generations", headers=headers, json=payload)
        if r.status_code != 200:
            return None
        data = r.json()
        try:
            b64 = data["data"][0].get("b64_json")
            if not b64:
                url = data["data"][0].get("url")
                return url if url else None
            return f"data:image/png;base64,{b64}"
        except Exception:
            return None


@app.post("/generate_story")
async def generate_story(params: StoryRequest):
    characters: List[str] = []
    if params.character_input:
        raw = params.character_input.strip()
        if raw.isdigit():
            # Ask the model to name them
            characters = []
        else:
            characters = [c.strip() for c in raw.split(",") if c.strip()]

    system_message = f"""
    You are a world-class interactive storyteller. You ALWAYS return strict JSON in the schema: \
    {"preface":string,"sections":[{"title":string,"summary":string,"prompt":string,"text":string}]} \
    Where `prompt` is a concise, vivid text-to-image prompt for this section (no more than 60 tokens). \
    Keep language age-appropriate for the audience and avoid copyrighted characters.
    """

    sys_prompt = (
        "You are a world-class interactive storyteller. You ALWAYS return strict JSON in the schema:\n"
        "{\n  \"preface\": string,\n  \"sections\": [ { \"title\": string, \"summary\": string, \"prompt\": string, \"text\": string } ]\n}\n"
        "Where `prompt` is a concise, vivid text-to-image prompt for this section (no more than 60 tokens).\n"
        "Keep language age-appropriate for the audience and avoid copyrighted characters."
    )

    user_prompt = {
        "genre": params.genre,
        "characters": characters if characters else params.character_input,
        "paragraphs": params.paragraphs,
        "audience": params.audience,
        "tone": params.tone,
        "requirements": [
            "Write exactly `paragraphs` sections.",
            "Each section must have: title, 1–3 sentence summary, an image prompt, and ~120–180 words of narrative text.",
            "Use consistent POV and tense.",
            "Provide a short summary / preface first.",
            "Separate each paragraph clearly with '---'"
        ],
        "if_characters_number": "If `characters` is a number, invent distinct names with short descriptors.",
    }

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": json.dumps(user_prompt)},
    ]

    content = await call_deepinfra_chat(messages, model=DEFAULT_LLM_MODEL, response_json=True)

    try:
        story = json.loads(content)
    except Exception:
        raise HTTPException(status_code=500, detail="Unable to generate story")


    # Optionally generate per-section images
    if params.include_images:
        for sec in story.get("sections", []):
            img_prompt = sec.get(
                "prompt") or f"{params.genre} illustration of: {sec.get('summary') or sec.get('title')}"
            img_data = await call_deepinfra_images(img_prompt, model=DEFAULT_IMAGE_MODEL)
            sec["image"] = img_data  # may be None if generation failed

    # Optionally add a short preface/summary
    if not params.include_preface:
        story.pop("preface", None)

    return { "image": DEFAULT_IMAGE_MODEL if params.include_images else None,
            "story": story}
