"""
Minimal conversational agent using OpenAI's native SDK.
No LangChain, no LangGraph - just clean, simple tool calling.
"""

import json
import os
from typing import AsyncGenerator, Dict, Any, List
import redis
from openai import AsyncOpenAI

from .tools import TOOL_DEFINITIONS, TOOL_FUNCTIONS
from .metadata import get_corpus_metadata

# Initialize
client = AsyncOpenAI()
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")


def build_system_prompt() -> str:
    """
    Build system prompt with dynamic metadata.
    Metadata is cached (1 hour TTL), so this is fast even with millions of rows.
    """
    meta = get_corpus_metadata()
    
    date_range = meta.get("date_range", {})
    speakers = meta.get("speakers", [])
    total = meta.get("total_utterances", 0)
    
    # Format temporal comparison dates
    temporal_guidance = ""
    if date_range.get("min") and date_range.get("max"):
        min_date = date_range["min"]
        max_date = date_range["max"]
        
        # Compute midpoint for splitting periods
        from datetime import datetime
        try:
            start = datetime.fromisoformat(min_date)
            end = datetime.fromisoformat(max_date)
            midpoint = start + (end - start) / 2
            mid_str = midpoint.date().isoformat()
            
            temporal_guidance = f"""
**Data coverage:** {min_date} to {max_date} ({total:,} utterances)

**For temporal questions** ("over time", "improved", "changed"), use compare_periods with:
- Early period: {min_date} to {mid_str}
- Late period: {mid_str} to {max_date}
"""
        except:
            temporal_guidance = f"**Data available:** {min_date} to {max_date}"
    
    speaker_list = ", ".join(speakers) if speakers else "various team members"
    
    return f"""You are a performance insights advisor for shop floor communication analysis.

{temporal_guidance}
**Team:** {speaker_list}

**Available metrics:**
- SAFETY_Score, QUALITY_Score, DELIVERY_Score, COST_Score, PEOPLE_Score (0-50 scale)
- comm_Pausing, comm_Clarifying_Questions, comm_Verbal_Affirmation (1-5 scale)

**How to answer:**
- **IMPORTANT:** If unsure about a speaker's name, use **list_speakers** first to check availability
- Use **search_utterances** to find what people said
- Use **get_metric_stats** for current stats (can filter by speaker/dates)
- Use **compare_periods** for temporal questions (use the periods above)
- If you get an error from a tool (e.g., "Speaker not found"), tell the user directly - don't guess or substitute
- Always cite specific numbers, dates, speakers
- Be conversational, proactive, helpful

**Context:** Remember the conversation. "What about Mike?" after asking about Tasha means Mike's data on the same metric.
"""


async def run_agent(
    question: str,
    session_id: str
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Run conversational agent with tool calling.
    Streams responses as they're generated.
    """
    
    # Load conversation history
    messages = await load_history(session_id)
    
    # Add user question
    messages.append({"role": "user", "content": question})
    
    # Agent loop (max 3 iterations to prevent infinite loops)
    iteration = 0
    max_iterations = 3
    
    while iteration < max_iterations:
        iteration += 1
        
        # Clean messages before sending to OpenAI (remove empty tool_calls)
        clean_messages = []
        for msg in messages:
            cleaned = {k: v for k, v in msg.items() if not (k == "tool_calls" and not v)}
            clean_messages.append(cleaned)
        
        print(f"🔍 Sending {len(clean_messages)} messages to OpenAI (iteration {iteration})")
        for i, msg in enumerate(clean_messages):
            has_tools = "tool_calls" in msg
            tc_val = msg.get("tool_calls") if has_tools else None
            print(f"  [{i}] role={msg.get('role')}, tool_calls={tc_val}, content_len={len(str(msg.get('content', '')))}")
        
        # Call OpenAI with tools
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=clean_messages,
                tools=TOOL_DEFINITIONS,
                stream=True
            )
        except Exception as e:
            print(f"❌ OpenAI API error: {e}")
            print(f"   Messages sent: {json.dumps(clean_messages, indent=2)}")
            raise
        
        # Collect response
        current_message = {"role": "assistant", "content": ""}
        tool_call_buffer = {}
        
        async for chunk in response:
            delta = chunk.choices[0].delta
            
            # Content streaming
            if delta.content:
                current_message["content"] += delta.content
                yield {"type": "token", "content": delta.content}
            
            # Tool call streaming
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_call_buffer:
                        tool_call_buffer[idx] = {
                            "id": tc.id or "",
                            "type": "function",
                            "function": {"name": tc.function.name or "", "arguments": ""}
                        }
                    
                    if tc.function.name:
                        tool_call_buffer[idx]["function"]["name"] = tc.function.name
                    if tc.function.arguments:
                        tool_call_buffer[idx]["function"]["arguments"] += tc.function.arguments
                    if tc.id:
                        tool_call_buffer[idx]["id"] = tc.id
        
        # Build clean assistant message (never include empty tool_calls)
        assistant_msg = {"role": "assistant", "content": current_message["content"]}
        if tool_call_buffer:
            assistant_msg["tool_calls"] = list(tool_call_buffer.values())
        
        # Add to history
        messages.append(assistant_msg)
        
        # Execute tools if any
        if tool_call_buffer:
            tool_calls_list = list(tool_call_buffer.values())
            yield {"type": "status", "message": f"🔍 Using {len(tool_calls_list)} tool(s)..."}
            
            for tool_call in tool_calls_list:
                func_name = tool_call["function"]["name"]
                func_args = json.loads(tool_call["function"]["arguments"])
                
                # Execute tool
                yield {"type": "status", "message": f"📊 Fetching {func_name}..."}
                
                if func_name in TOOL_FUNCTIONS:
                    result = TOOL_FUNCTIONS[func_name](**func_args)
                else:
                    result = {"error": f"Unknown tool: {func_name}"}
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(result)
                })
            
            # Continue loop to let LLM synthesize
            continue
        
        # No more tool calls, we're done
        break
    
    # Save conversation
    await save_history(session_id, messages)
    
    # Final message
    final_answer = current_message.get("content", "")
    yield {
        "type": "final",
        "answer": final_answer,
        "tool_calls_made": iteration - 1
    }


async def load_history(session_id: str) -> List[Dict[str, str]]:
    """Load conversation history from Redis"""
    # Build system prompt with current metadata
    system_prompt = build_system_prompt()
    
    if not session_id:
        return [{"role": "system", "content": system_prompt}]
    
    try:
        r = redis.from_url(REDIS_URL)
        raw = r.get(f"agent:history:{session_id}")
        
        if raw:
            history = json.loads(raw)
            
            # Clean up messages - remove empty tool_calls arrays
            cleaned_history = []
            for msg in history:
                cleaned_msg = {k: v for k, v in msg.items() if k != "tool_calls" or (k == "tool_calls" and v)}
                cleaned_history.append(cleaned_msg)
            
            # Replace system prompt with fresh one (has latest metadata)
            if cleaned_history and cleaned_history[0].get("role") == "system":
                cleaned_history[0] = {"role": "system", "content": system_prompt}
            else:
                cleaned_history = [{"role": "system", "content": system_prompt}] + cleaned_history
            
            return cleaned_history
        
    except Exception as e:
        print(f"Failed to load history: {e}")
    
    return [{"role": "system", "content": system_prompt}]


async def save_history(session_id: str, messages: List[Dict[str, str]]):
    """Save conversation history to Redis - only user/assistant pairs, not tool calls"""
    if not session_id:
        return
    
    try:
        r = redis.from_url(REDIS_URL)
        
        # Keep only system, user, and assistant messages (skip tool messages)
        # And remove any tool_calls from assistant messages
        clean_messages = []
        for msg in messages:
            role = msg.get("role")
            
            if role in ["system", "user"]:
                clean_messages.append(msg)
            elif role == "assistant":
                # Only save content, never tool_calls
                clean_messages.append({
                    "role": "assistant",
                    "content": msg.get("content", "")
                })
            # Skip "tool" role messages entirely
        
        # Keep last 20 messages
        recent = clean_messages[-20:]
        r.set(f"agent:history:{session_id}", json.dumps(recent))
    except Exception as e:
        print(f"Failed to save history: {e}")
