#!/bin/bash
# Test chat endpoint and save results to file
OUTPUT_FILE="/home/btech01_06/Desktop/DeepLearning/Cortex-Lab/test_results.txt"
echo "=== Chat Endpoint Tests ===" > "$OUTPUT_FILE"
echo "Started at: $(date)" >> "$OUTPUT_FILE"

echo "" >> "$OUTPUT_FILE"
echo "=== Test 1: Simple Hello ===" >> "$OUTPUT_FILE"
RESPONSE=$(curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}],"temperature":0.6,"max_tokens":256,"stream":false}' 2>&1)

echo "RAW JSON:" >> "$OUTPUT_FILE"
echo "$RESPONSE" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "$RESPONSE" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print('THINKING:', repr(d.get('thinking','')[:300]))
    print('CONTENT:', repr(d.get('content','')[:500]))
    print('Has User:?', 'User:' in d.get('content',''))
    print('Content length:', len(d.get('content','')))
except Exception as e:
    print('PARSE ERROR:', e)
" >> "$OUTPUT_FILE" 2>&1

echo "" >> "$OUTPUT_FILE"
echo "=== Test 2: Personal Question ===" >> "$OUTPUT_FILE"
RESPONSE2=$(curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"what is my name?"}],"temperature":0.6,"max_tokens":256,"stream":false}' 2>&1)

echo "RAW JSON:" >> "$OUTPUT_FILE"
echo "$RESPONSE2" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "$RESPONSE2" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print('THINKING:', repr(d.get('thinking','')[:300]))
    print('CONTENT:', repr(d.get('content','')[:500]))
    print('Has User:?', 'User:' in d.get('content',''))
    print('Content length:', len(d.get('content','')))
except Exception as e:
    print('PARSE ERROR:', e)
" >> "$OUTPUT_FILE" 2>&1

echo "" >> "$OUTPUT_FILE"
echo "=== DONE ===" >> "$OUTPUT_FILE"
echo "Finished at: $(date)" >> "$OUTPUT_FILE"
echo "TESTS_COMPLETE"
