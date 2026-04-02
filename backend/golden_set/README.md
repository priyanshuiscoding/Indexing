# Golden Set Format

Place one JSON file per PDF in this folder.

Required fields:
- `pdf_id`: the ingested PDF id
- `expected_index`: array of expected TOC rows

Example:
```json
{
  "pdf_id": "155edf7f8b4a528c",
  "expected_index": [
    {"title": "Memo of bail application", "pageFrom": 2, "pageTo": 8},
    {"title": "A copy impugned Trial court order", "pageFrom": 9, "pageTo": 11}
  ]
}
```

Use endpoint:
- `GET /api/golden-eval?limit=50`
