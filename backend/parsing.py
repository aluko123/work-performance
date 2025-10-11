import io
import json
import asyncio
from typing import List, Dict, Any
from openai import AsyncOpenAI
from unstructured.partition.auto import partition
from .prompts import (
    single_shot_system,
    single_shot_user,
    chunk_system,
)

async def extract_transcript_single_shot(raw_text: str, client: AsyncOpenAI) -> list:
    """
    Uses a single, large OpenAPI API call to extract structured transcript data.
    Works for moderately sized documents.
    """
    print("Extracting transcript using 'single_shot' strategy...")
    system_prompt = single_shot_system()

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": single_shot_user(raw_text)},
            ],
            response_format={"type": "json_object"},
            max_tokens=16384
        )
        response_content = response.choices[0].message.content
        
        # Log the raw response for debugging purposes
        print("--- Raw OpenAI Response (single_shot) ---")
        print(response_content)
        print("----------------------------------------")

        extracted_data = json.loads(response_content)
        return extracted_data.get("transcript", [])
        
    except json.JSONDecodeError as e:
        print(f"JSON DECODE ERROR: The AI's response was not valid JSON. Error: {e}")
        print("The raw response that caused the error is printed above.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred with the OpenAI API call: {e}")
        return []


# --- Strategy 2: Chunking for large documents ---

def _split_text(text: str, chunk_size: int = 8000, overlap: int = 200) -> List[str]:
    """Splits text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

async def _process_chunk(chunk: str, client: AsyncOpenAI) -> List[Dict[str, Any]]:
    """Helper function to process a single text chunk with OpenAI."""
    system_prompt = chunk_system()
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk}
            ],
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        return data.get("utterances", [])
    except Exception as e:
        print(f"Error processing a chunk: {e}")
        return []

async def extract_transcript_chunking(raw_text: str, client: AsyncOpenAI) -> list:
    """
    Uses a chunking and parallel processing strategy to extract transcript data.
    Ideal for very large documents and avoiding rate limits.
    """
    print("Extracting transcript using 'chunking' strategy...")
    chunks = _split_text(raw_text)
    tasks = [_process_chunk(chunk, client) for chunk in chunks]
    
    results_from_chunks = await asyncio.gather(*tasks)
    
    # Flatten the list of lists into a single list of utterances
    full_transcript = [item for sublist in results_from_chunks for item in sublist]
    
    # Post-processing to find a single date could be added here if needed
    # For now, we are omitting the date from the chunked results.
    return full_transcript

def parse_document_with_unstructured(file_content: bytes, content_type: str) -> str:
    """
    Uses 'unstructured' library to parse the raw content of an uploaded file.
    Handles PDF, XLSX, TXT, etc. all in one.
    """
    print(f"Parsing document with content_type '{content_type}' using unstructured")
    try:
        #partition file into element chunks
        elements = partition(file=io.BytesIO(file_content), content_type=content_type)
        #join the extracted elements into a string
        return "\n\n".join([str(el) for el in elements])
    except Exception as e:
        print(f"Error parsing document with unstructured: {e}")
        return ""
