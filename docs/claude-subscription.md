# Translating with your Claude subscription

The `claude_cli` provider runs the [Claude Code](https://docs.claude.com/en/docs/claude-code) CLI in non-interactive mode. Translation then uses your Claude plan's quota instead of a paid API key. It serves three ops:

| Op | When | What |
|---|---|---|
| `mt.context` | once per video | summary, glossary, cast sheet (gender, age, relations, forms of address), open questions |
| `mt.translate` | once per window of ~12 lines | `{short, target, long}` for every line + the listener and address terms used |
| `mt.rewrite` | profile `final`, only for lines that are still too long | a shorter line |

## The command it runs

```bash
claude -p --model sonnet --max-turns 2 --output-format json --tools "" \
       --strict-mcp-config --no-session-persistence \
       --system-prompt "<configs/prompts/...md>" --json-schema '<schema>'   # prompt on stdin
```

The validated JSON is read from the `structured_output` field of the result.

- `--tools ""`: no tools, the model can only answer.
- `--max-turns 2`: structured output counts as an extra turn in `num_turns`.
- `--json-schema`: the schemas are flat and fully required, generated from the pydantic models in `src/dub/providers/claude_cli.py`.

## On the laptop

If `claude` works in your terminal, nothing else is needed:

```bash
uv run dub run demo/interview.mp4 --to vi --fake --use "mt.*=claude_cli"
```

## On a GPU pod (M1)

1. On the laptop, once: `claude setup-token`. It prints a long-lived token for model calls only.
2. Store it as a RunPod secret and expose it as `CLAUDE_CODE_OAUTH_TOKEN` in the pod.
3. Install the CLI in the pod image.

## Two traps

- **`ANTHROPIC_API_KEY`:** if it is set, `claude` bills that API key instead of your subscription. The provider removes it from the child process environment and logs a note.
- **`--bare`:** this mode does not read the subscription login. The provider never uses it.

## Quota and errors

- The plan's quota is per session and per week, shared by all models.
  - `sonnet` is enough for translation.
  - In the demo, one 1-minute video took 3 calls (about 36 s in total).
- When the CLI reports a usage limit, the call is classified as `usage_limit`:
  1. the provider waits 60 s and retries, then waits 300 s and retries;
  2. after that the job stops.
  - Finished windows are cached. Rerun later and only the missing windows are sent.
- If you add an API-based provider (`llm_api`, not part of M0), it is next in the `defaults` chain and is used automatically as a fallback.

## Personal use only

Using a subscription login inside an app that other people use is not allowed. If you ever share this app, switch the `mt.*` route to an API-key provider. That is a one-line change in `dub.yaml` or a `--use` flag.
