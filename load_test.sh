#!/bin/bash

# Load Test Script for Profiling ML Inference
# This script hits inference-heavy endpoints to profile BERT, embeddings, and RAG

echo "🔥 Starting inference load test..."
echo "Make sure profiling is running in another terminal:"
echo "  docker compose exec backend py-spy record -o /app/data/inference_profile.svg --duration 120 --pid 1"
echo ""
sleep 3

BASE_URL="http://localhost:8000"

# Test 1: File upload (triggers BERT inference + embeddings)
echo "📄 Testing file upload endpoint (BERT + embeddings)..."
if [ -f "sample_meeting.txt" ]; then
    for i in {1..5}; do
        echo "  Upload #$i"
        curl -s -F "text_file=@sample_meeting.txt" $BASE_URL/analyze_text/ > /dev/null &
    done
else
    echo "  ⚠️  sample_meeting.txt not found, skipping"
fi

sleep 2

# Test 2: RAG/Insights endpoint (ChromaDB + OpenAI)
echo "🤖 Testing RAG insights endpoint..."
for i in {1..10}; do
    curl -s -X POST "$BASE_URL/api/get_insights" \
        -H "Content-Type: application/json" \
        -d '{"question":"What are the main performance trends?"}' > /dev/null &
done

sleep 2

# Test 3: Trends endpoint (DB queries + aggregation)
echo "📊 Testing trends endpoint..."
for metric in comm_clarity sa_performance deviation_behavior; do
    for period in daily weekly monthly; do
        curl -s "$BASE_URL/api/trends?metric=$metric&period=$period" > /dev/null &
    done
done

sleep 2

# Test 4: List analyses (DB queries)
echo "📋 Testing analyses list endpoint..."
for i in {1..20}; do
    curl -s "$BASE_URL/analyses/" > /dev/null &
done

echo ""
echo "✅ Load test complete! Wait for all requests to finish..."
wait
echo "🔍 Check ./data/inference_profile.svg for results"
