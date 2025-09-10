import io
import json
import asyncio
from typing import List, Dict, Optional
from unstructured.partition.auto import partition
from chunkr_ai import Chunkr
from openai import AsyncOpenAI
from datetime import datetime
import re
from . import utils

class RobustMeetingExtractor:
    def __init__(self, chunkr_api_key: Optional[str], openai_client: AsyncOpenAI):
        self.llm_client = openai_client
        self.max_tokens_per_llm_call = 2000
        if chunkr_api_key:
            self.chunkr = Chunkr(api_key=chunkr_api_key)
            print("Chunkr client initialized.")
        else:
            self.chunkr = None
            print("Chunkr API key not found. The extractor will use the fallback chunking method.")

    async def process_any_document(self, file_path: str, original_filename: str) -> Dict:
        """Three-stage pipeline for any document format"""
        print(f"Starting processing for {original_filename}...")
        raw_elements = self.extract_with_unstructured(file_path)
        
        if self.chunkr:
            try:
                print("Executing primary 3-stage pipeline with Chunkr...")
                structured_chunks = await self.enhance_with_chunkr(file_path, raw_elements)
                # The global context step isn't strictly necessary for the Chunkr path but could be added for more robustness.
                meeting_data = await self.extract_meeting_data_with_llm(structured_chunks)
                return self.format_for_database(meeting_data, stages=["unstructured", "chunkr", "llm"])
            except Exception as e:
                print(f"Primary 3-stage pipeline failed: {e}. Falling back to 2-stage process.")
                return await self.fallback_processing(file_path, raw_elements)
        else:
            return await self.fallback_processing(file_path, raw_elements)

    def extract_with_unstructured(self, file_path: str) -> List:
        print("Stage 1: Extracting elements with Unstructured...")
        try:
            return partition(filename=file_path)
        except Exception as e:
            print(f"Unstructured failed: {e}")
            return []

    async def enhance_with_chunkr(self, file_path: str, unstructured_elements: List) -> List[Dict]:
        print("Stage 2: Enhancing with Chunkr for semantic chunking...")
        chunkr_result = await self.chunkr.upload(file_path)
        return self.merge_processing_results(unstructured_elements, chunkr_result)

    def merge_processing_results(self, unstructured_elements, chunkr_result) -> List[Dict]:
        print("Merging Unstructured and Chunkr results...")
        chunks = []
        for chunkr_chunk in chunkr_result.chunks:
            chunks.append({"content": chunkr_chunk.content, "type": chunkr_chunk.type})
        return chunks

    async def fallback_processing(self, file_path: str, elements: List) -> Dict:
        """Intelligent fallback using global context extraction."""
        print("Executing intelligent fallback processing...")
        global_context = await self._extract_global_context(elements)
        chunks = self.create_chunks_from_unstructured(elements)
        meeting_data = await self.extract_meeting_data_with_llm(chunks, global_context)
        processed_data = self._fill_forward_dates(meeting_data)
        return self.format_for_database(processed_data, stages=["unstructured", "llm_with_context"])

    async def _extract_global_context(self, elements: List) -> Dict:
        print("Extracting global context from document header...")
        header_text = "\n".join([str(el) for el in elements[:10]])
        prompt = f'''From the following text, which is the header of a document, extract the single, primary date of the meeting. Return JSON with one key, 'meeting_date'. If no date is found, return null.\n\nTEXT: {header_text}'''
        try:
            response = await self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a date extraction expert."}, 
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            return {"date": data.get("meeting_date")}
        except Exception as e:
            print(f"Could not extract global date: {e}")
            return {"date": None}

    def create_chunks_from_unstructured(self, elements) -> List[Dict]:
        print("Creating chunks from Unstructured elements (fallback)...")
        chunks, current_chunk, current_tokens = [], [], 0
        for element in elements:
            element_text = str(element)
            tokens = len(element_text.split())
            if current_tokens + tokens > self.max_tokens_per_llm_call:
                if current_chunk:
                    chunks.append({"content": "\n".join(current_chunk)})
                current_chunk, current_tokens = [element_text], tokens
            else:
                current_chunk.append(element_text)
                current_tokens += tokens
        if current_chunk:
            chunks.append({"content": "\n".join(current_chunk)})
        return chunks

    async def extract_meeting_data_with_llm(self, chunks: List[Dict], global_context: Optional[Dict] = None) -> List[Dict]:
        print(f"Stage 3: Extracting meeting data from {len(chunks)} chunks using LLM...")
        tasks = [self.process_single_chunk_with_llm(chunk, i, global_context) for i, chunk in enumerate(chunks)]
        list_of_results = await asyncio.gather(*tasks)
        all_extractions = [item for sublist in list_of_results for item in sublist]
        return self.deduplicate_and_clean(all_extractions)

    async def process_single_chunk_with_llm(self, chunk: Dict, i: int, global_context: Optional[Dict]) -> List[Dict]:
        try:
            prompt = self.build_adaptive_prompt(chunk, i, global_context)
            extraction = await self.llm_extract_with_retry(prompt)
            # Standardize date and time right after extraction
            for item in extraction:
                item['date'] = utils.parse_date(item.get('date'))
                item['timestamp'] = utils.parse_time(item.get('timestamp'))
            return extraction if extraction else []
        except Exception as e:
            print(f"LLM extraction failed for chunk {i}: {e}")
            return []

    def build_adaptive_prompt(self, chunk: Dict, chunk_id: int, global_context: Optional[Dict]) -> str:
        global_date_hint = global_context.get('date') if global_context else None
        date_prompt = f"The primary date for the entire document is likely {global_date_hint}. " if global_date_hint else ""
        date_prompt += "For each utterance, determine its specific date. If a date is mentioned in the chunk, use it. If no date is mentioned, use the primary document date if available, otherwise use null."
        
        base_prompt = f"""{date_prompt}\n\nCHUNK {chunk_id}: Extract meeting data from this text.\n\nCONTENT:\n{chunk['content'][:8000]}\n\nLook for: Timestamps, Speaker names, and what each person said (utterances)."""
        base_prompt += "\n\nReturn JSON array ONLY of objects with keys \"date\", \"timestamp\", \"speaker\", \"utterance\"."
        return base_prompt

    async def llm_extract_with_retry(self, prompt: str, max_retries: int = 2) -> List[Dict]:
        for attempt in range(max_retries):
            try:
                response = await self.llm_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert data extractor. Output JSON only."}, 
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                response_content = response.choices[0].message.content
                data = json.loads(response_content)
                for key, value in data.items():
                    if isinstance(value, list):
                        return value
                return []
            except Exception as e:
                print(f"LLM/JSON error attempt {attempt + 1}: {e}")
                await asyncio.sleep(1)
        return []

    def _fill_forward_dates(self, extractions: List[Dict]) -> List[Dict]:
        print("Post-processing: Filling forward missing dates...")
        last_seen_date = None
        for item in extractions:
            if item.get('date'):
                last_seen_date = item['date']
            elif last_seen_date:
                item['date'] = last_seen_date
        return extractions

    def deduplicate_and_clean(self, extractions: List[Dict]) -> List[Dict]:
        seen = set()
        deduplicated = []
        for item in extractions:
            identifier = (item.get('speaker'), item.get('utterance'))
            if identifier not in seen:
                seen.add(identifier)
                deduplicated.append(item)
        return deduplicated

    def format_for_database(self, meeting_data: List[Dict], stages: List[str]) -> Dict:
        return {
            "status": "success",
            "extractions": meeting_data,
            "processing_metadata": {
                "stages_completed": stages,
                "extraction_timestamp": datetime.now().isoformat()
            }
        }
