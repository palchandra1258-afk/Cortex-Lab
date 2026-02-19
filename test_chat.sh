#!/bin/bash
echo "=== Test 1: Simple Hello ==="
RESPONSE=$(curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}],"temperature":0.6,"max_tokens":256,"stream":false}')

echo "$RESPONSE" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('THINKING:', repr(d.get('thinking','')[:300]))
print()
print('CONTENT:', repr(d.get('content','')[:500]))
print()
print('Has User:?', 'User:' in d.get('content',''))
print('Content length:', len(d.get('content','')))
"

echo ""
echo "=== Test 2: Personal Question (should NOT fabricate) ==="
RESPONSE2=$(curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"what is my name?"}],"temperature":0.6,"max_tokens":256,"stream":false}')

echo "$RESPONSE2" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('THINKING:', repr(d.get('thinking','')[:300]))
print()
print('CONTENT:', repr(d.get('content','')[:500]))
print()
print('Has User:?', 'User:' in d.get('content',''))
print('Content length:', len(d.get('content','')))
"
