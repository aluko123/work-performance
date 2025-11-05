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
from . import metrics as metrics_registry

# Initialize
client = AsyncOpenAI()
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")
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
    
    # Build dynamic metric lists for the prompt (keep concise)
    cats = metrics_registry.split_metrics()
    agg_list = ", ".join(cats.get("aggregated", [])[:10]) or "SAFETY_Score, QUALITY_Score, DELIVERY_Score, COST_Score, PEOPLE_Score"
    gran_list = ", ".join(cats.get("granular", [])[:20]) or "comm_Clarifying_Questions, comm_Pausing, feedback_Timely, ..."

    return f"""You are a performance insights advisor for shop floor communication analysis.

{temporal_guidance}
**Team:** {speaker_list}

**Available metrics:**
- Aggregates: {agg_list}
- Behaviors (sample): {gran_list}
- Use the list_metrics tool to browse the full set with human-friendly names.

**How to answer:**
- **IMPORTANT:** If unsure about a speaker's name, use **list_speakers** first to check availability
- **When users ask "what did X say" or "what was discussed"**, ALWAYS use **search_utterances** to find actual quotes
- When selecting metrics, use **suggest_metrics** to pick the most relevant ones
- Use **get_metric_stats** for current stats (can filter by speaker/dates)
- Use **compare_periods** for temporal questions (use the periods above)

**Aggregates vs. Behaviors:**
- For aggregate or overall questions (e.g., "overall performance/safety/quality/people"), favor aggregate metrics (e.g., SAFETY_Score, QUALITY_Score, PEOPLE_Score).
- For behavior/communication questions, select the most relevant behavior metrics and you may combine multiple metrics (e.g., comm_Clarifying_Questions + comm_Probing_Questions). Use grouped_bar charts to compare multiple metrics.

**Metric Inference:**
When users don't specify a metric:
- "behavior", "performance" → Use PEOPLE_Score (overall)
- "communication" → Use comm_Clarifying_Questions or Total_Comm_Score
- "safety" → Use SAFETY_Score
- When unclear, default to PEOPLE_Score and mention which metric you chose

**Chart Generation - MANDATORY (DO NOT SKIP):**
You MUST generate a chart for every trend/comparison question. This is non-negotiable.

1. **Trend questions** ("improved", "changed", "over time", "trends", "behavior over time"):
   - Call compare_periods or get_metric_stats
   - IMMEDIATELY after, ALWAYS call: generate_chart(chart_type="line", metric=X, speaker=Y if specified)
   
2. **Comparison questions** ("compare", "vs", "versus", ANY question comparing 2+ people):
   - Step 1: Call get_metric_stats for each speaker
   - Step 2 (REQUIRED): Call generate_chart(chart_type="bar", metric=X)
   - Example: "Compare A and B on safety" → get_metric_stats(SAFETY_Score, A), get_metric_stats(SAFETY_Score, B), then generate_chart(bar, SAFETY_Score)
   - Never finish a comparison without calling generate_chart
3. **Multi-metric comparisons** ("compare multiple behaviors"):
   - Use generate_chart(chart_type="grouped_bar", metrics=[M1, M2, ...], speaker=Y if specified)

**Citations - REQUIRED:**
For questions about specific speakers ("Tasha's behavior", "what Rosa said"):
- ALWAYS call: search_utterances(speaker=X, query="relevant topic", top_k=3)
- Include 2-3 short quotes in your answer

**Response Format for Chat:**
- Start with 3-5 SHORT bullets (plain text, NO markdown bold/italic)
- Then 1-2 sentences of context
- Be concise and scannable
- NEVER include markdown images (![...])
- Always cite specific numbers, dates
- If error, tell user directly

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
                model=GPT_MODEL,
                messages=clean_messages,
                tools=TOOL_DEFINITIONS,
                temperature=0.2,  # Lower temp for more consistent, deterministic responses
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
            
            # Track what tools were called for chart enforcement
            called_tools = set()
            metric_used = None
            speakers_queried = []
            trend_speakers = []  # Track speakers for trend queries
            
            for tool_call in tool_calls_list:
                func_name = tool_call["function"]["name"]
                func_args = json.loads(tool_call["function"]["arguments"])
                called_tools.add(func_name)
                
                # Track metric and speakers for auto-chart generation
                if func_name == "get_metric_stats":
                    metric_used = func_args.get("metric")
                    if "speaker" in func_args:
                        speakers_queried.append(func_args["speaker"])
                elif func_name == "compare_periods":
                    metric_used = func_args.get("metric")
                    # Track speakers for trend line charts (supports multiple)
                    if "speaker" in func_args:
                        trend_speakers.append(func_args["speaker"])
                
                # Execute tool
                yield {"type": "status", "message": f"📊 Fetching {func_name}..."}
                
                if func_name in TOOL_FUNCTIONS:
                    result = TOOL_FUNCTIONS[func_name](**func_args)
                else:
                    result = {"error": f"Unknown tool: {func_name}"}
                
                # Emit tool result for main.py to process
                yield {"type": "tool_result", "tool_name": func_name, "result": result}
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(result)
                })
            
            # 🔒 ENFORCE CHART GENERATION (programmatic fail-safe)
            # If LLM called comparison/trend tools but forgot to call generate_chart, inject it
            should_have_chart = False
            chart_type = None
            chart_reason = None
            
            # Detect comparison query (multiple speakers queried)
            if "get_metric_stats" in called_tools and len(speakers_queried) >= 2:
                should_have_chart = True
                chart_type = "bar"
                chart_reason = f"comparison of {len(speakers_queried)} speakers on {metric_used}"
            
            # Detect trend query (compare_periods called)
            elif "compare_periods" in called_tools:
                should_have_chart = True
                chart_type = "line"
                if trend_speakers:
                    speaker_count = len(trend_speakers)
                    speaker_msg = f" for {speaker_count} speaker{'s' if speaker_count > 1 else ''}"
                else:
                    speaker_msg = ""
                chart_reason = f"trend analysis for {metric_used}{speaker_msg}"
            
            # Auto-inject chart if needed and not already called
            if should_have_chart and "generate_chart" not in called_tools and metric_used:
                print(f"🔒 Auto-injecting chart generation (LLM forgot): {chart_reason}")
                yield {"type": "status", "message": f"📊 Generating chart for {chart_reason}..."}
                
                # Build chart arguments
                chart_args = {"chart_type": chart_type, "metric": metric_used}
                
                # For bar charts (comparisons), pass the specific speakers being compared
                if chart_type == "bar" and speakers_queried:
                    chart_args["speakers"] = speakers_queried
                
                # For line charts (trends), pass speaker filter(s)
                if chart_type == "line" and trend_speakers:
                    if len(trend_speakers) == 1:
                        # Single speaker - use speaker filter
                        chart_args["speaker"] = trend_speakers[0]
                    else:
                        # Multiple speakers - use speakers list to show all on same chart
                        chart_args["speakers"] = trend_speakers
                
                # Execute chart generation
                chart_result = TOOL_FUNCTIONS["generate_chart"](**chart_args)
                
                # Emit chart result
                yield {"type": "tool_result", "tool_name": "generate_chart", "result": chart_result}
                
                # Add synthetic tool call and result to messages
                # This makes the LLM aware a chart was generated
                synthetic_tool_call_id = f"auto_chart_{iteration}"
                
                # Add assistant message with the auto-injected tool call
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": synthetic_tool_call_id,
                        "type": "function",
                        "function": {
                            "name": "generate_chart",
                            "arguments": json.dumps(chart_args)
                        }
                    }]
                })
                
                # Add tool result
                messages.append({
                    "role": "tool",
                    "tool_call_id": synthetic_tool_call_id,
                    "content": json.dumps(chart_result)
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
