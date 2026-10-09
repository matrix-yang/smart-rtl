# LLM Routing

External LLM calls are disabled unless the user explicitly assigns a model to a
role.

Before any call, assemble the prompt from the allowed input set in
`agent-workflow.md`. Do not pass private scorer files, real test files,
expected-output code, mutation definitions, harness internals, or raw logs that
reveal private test structure.

Examples:

- no assignment: all roles run as coding sub-agents
- `generator=nvidia/nemotron-3-ultra-550b-a55b`: only generator calls that model
- `generator=<model>, reflector=<model>, coordinator=<model>`: each assigned
  role uses its model

Default hosted provider:

- provider: NVIDIA Inference API
- endpoint: `https://integrate.api.nvidia.com/v1`
- default model: `nvidia/nemotron-3-ultra-550b-a55b`
- key env var: `NVIDIA_API_KEY`
- default output budget: `192000`
- reasoning: enabled by default with `ACE_RTL_ENABLE_REASONING=1`, sent through
  `extra_body={"chat_template_kwargs": {"enable_thinking": true}, "reasoning_budget": 192000}`
- streaming: enabled by default with `ACE_RTL_STREAM=1`
- sampling defaults: `ACE_RTL_TEMPERATURE=0.2`, `ACE_RTL_TOP_P=0.95`

Do not print keys in logs or reports.

Call:

```bash
python3 skills/ace-rtl/scripts/llm_call.py \
  --component generator \
  --model nvidia/nemotron-3-ultra-550b-a55b \
  --provider nvidia \
  --system "You are a helpful SystemVerilog design assistant." \
  --prompt-file /path/to/prompt.txt \
  --output /path/to/response.json
```

Custom adapter:

```bash
export ACE_RTL_LLM_SCRIPT=/path/to/your_llm_adapter.py
python3 skills/ace-rtl/scripts/llm_call.py --provider custom ...
```

The adapter receives JSON on stdin:

```json
{"model": "<model>", "messages": [], "temperature": 0.2, "top_p": 0.95, "max_tokens": 192000, "stream": true, "extra_body": {"chat_template_kwargs": {"enable_thinking": true}, "reasoning_budget": 192000}}
```

It may return raw assistant text, `{"content": "..."}`, or an
chat-completions-style `choices[0].message.content` payload.

Parse:

```bash
python3 skills/ace-rtl/scripts/extract_solution.py \
  --input /path/to/response.json \
  --output /path/to/generated.sv
```

For multi-file target sections:

```bash
python3 skills/ace-rtl/scripts/extract_solution.py \
  --input /path/to/response.json \
  --target-dir /path/to/generated_targets
```
