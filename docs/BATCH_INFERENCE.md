# Batch Inference API

Hippo v0.2.0+ supports batch inference endpoints for processing multiple requests in parallel.

## Endpoints

### POST /api/batch/generate

Batch completion generation (non-streaming).

**Request:**
```json
{
  "requests": [
    {"model": "llama3", "prompt": "Hello", "options": {"temperature": 0.7}},
    {"model": "llama3", "prompt": "World", "options": {"temperature": 0.5}}
  ],
  "max_concurrent": 2
}
```

**Response:**
```json
{
  "results": [
    {"response": "...", "status": "success"},
    {"response": "...", "status": "success"}
  ],
  "total_duration_ns": 123456789,
  "success_count": 2,
  "error_count": 0
}
```

### POST /api/batch/chat

Batch chat completion (non-streaming).

**Request:**
```json
{
  "requests": [
    {"model": "llama3", "messages": [{"role": "user", "content": "Hi"}]},
    {"model": "llama3", "messages": [{"role": "user", "content": "Bye"}]}
  ],
  "max_concurrent": 2
}
```

**Response:**
```json
{
  "results": [
    {"response": {"role": "assistant", "content": "..."}, "status": "success"},
    {"response": {"role": "assistant", "content": "..."}, "status": "success"}
  ],
  "total_duration_ns": 123456789,
  "success_count": 2,
  "error_count": 0
}
```

## Configuration

Batch inference can be configured via YAML or environment variables:

```yaml
# ~/.hippo/config.yaml
batch:
  max_size: 32          # Maximum batch size
  max_concurrent: 4     # Maximum concurrent requests per batch
  timeout_seconds: 120  # Per-request timeout
```

**Environment variables:**
- `HIPPO_BATCH_MAX_SIZE`: Override batch max size (default: 32)

## Features

- **Concurrency Control**: Uses `asyncio.Semaphore` to limit concurrent requests
- **Error Isolation**: Single request failures don't affect others
- **Timeout Protection**: Each request has independent timeout (default: 120s)
- **Authentication**: Requires valid API key (if configured)
- **Metrics Integration**: Records request counts and durations

## Usage Example

```bash
# Set API key (if required)
export HIPPO_API_KEY="your-secret-key"

# Batch generate
curl -X POST http://localhost:11434/api/batch/generate \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"model": "llama3", "prompt": "What is 2+2?"},
      {"model": "llama3", "prompt": "What is 3+3?"}
    ],
    "max_concurrent": 2
  }'

# Batch chat
curl -X POST http://localhost:11434/api/batch/chat \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"model": "llama3", "messages": [{"role": "user", "content": "Hi"}]},
      {"model": "llama3", "messages": [{"role": "user", "content": "Hello"}]}
    ],
    "max_concurrent": 2
  }'
```

## Limits

- **Max batch size**: 32 (configurable)
- **Max concurrent**: 16 (hard limit for safety)
- **Per-request timeout**: 120s (configurable via `options.timeout`)

## Error Handling

Each result includes:
- `status`: "success" or "error"
- `response`: Generated content (if successful)
- `error`: Error message (if failed)

Example error result:
```json
{
  "response": "",
  "status": "error",
  "error": "Model 'xyz' not found in /Users/user/.hippo/models"
}
```
