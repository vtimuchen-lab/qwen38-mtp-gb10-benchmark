# Qwen3.8 MTP7 prefix-cache agent-session A/B

- Static system prefix: 18243 raw tokens
- 12 progressive tool-calling turns; 24 requests per mode; temperature 0, seed 42.

| Mode | Quality | Prompt evaluated | Prompt seconds | Decode tok/s | Wall seconds |
|---|---:|---:|---:|---:|---:|
| cache off | 12/12 | 460436 | 703.17 | 34.86 | 734.84 |
| cache on | 12/12 | 39381 | 72.11 | 35.12 | 103.10 |

- End-to-end wall speedup: **7.13x**.
- Prompt-processing time reduction: **89.74%**.
- Quality identical and fully passed: **True**.
