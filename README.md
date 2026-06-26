# raphael-ai

AI understanding, copilot, agents, workflow generation

## API

- Prefix: `/v1/ai`
- Port: `8099`
- Health: `GET /health`

## Events

_Published and consumed events documented in `openapi.yaml` and raphael-contracts._

## Development

```bash
uv sync
uv run uvicorn raphael_ai.app:app --reload --port 8099
```

Part of the [Raphael Platform](https://github.com/hummingbird-labs) by HummingBird Labs.
