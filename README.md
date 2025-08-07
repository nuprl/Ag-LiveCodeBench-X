# Ag-LiveCodeBench-X

This repository contains scripts used in the [Agnostics project](https://agnostics.abgru.me)
to evaluate models on Ag-LiveCodeBench-X,
a multi-PL variant of LiveCodeBench which is more of a challenge than MultiPL-E.

You can find out more about the Agnostics project, including the related artifacts,
on [its website](https://agnostics.abgru.me).

The scripts in this repository are meant to be run with `uv`.
You can execute the script directly or run them with `uv run $SCRIPT`.
(If you prefer not to use `uv`, the scripts start with a comment listing their dependencies.)

Here is a simple usage example:

```
uv run livecodebench_x.py \
    completions \
    --model-name openai/qwen3_8b_awq \
    --completions-path completions.jsonl \
    --temperature 0.2 \
    --num-concurrent 50 \
    --max-tokens 2048 \
    --language "Lua /nothink"

uv run livecodebench_x.py executions \
    --container-name ghcr.io/nuprl/agnostics:lua \
    --timeout-seconds 15 \
    --generations-path completions.jsonl \
    --executions-path executions.jsonl \
    --num-concurrent 50

uv run livecodebench_x.py pass1 executions.jsonl
```

## `livecodebench_x.py`
This script runs Ag-LiveCodeBench-X. To understand how this relates to LiveCodeBench,
you can read `prepare_lcbx.py`. But you will not need to run that script:
a prepared dataset is already available on the Hugging Face Hub.

You can use this script to evaluate LiveCodeBench-X on any programming language.
You can use one of the Agnostics verifier containers, or you can write your own.
The container must follow the Agnostics protocol, which communicates over JSON on
the standard input and output.

An input *line* has the shape:

```
{ "code": str, "timeout_s": int, "test_cases": [ { "input": str, "output": str }, ... ] }
```

An output *line* has the shape:

```
{ result: 'success'; stderr: string } |
{ result: 'fail:wrong-output'; expected: string; got: string; stderr: string } |
{ result: 'fail:error'; exit_code: number; stdout: string; stderr: string } |
{ result: 'fail:timeout'; stdout: string; stderr: string } |
{ result: 'fail:other'; stdout: string; stderr: string }
```

The code will only check for the "success" value and record the entire line of output.
So, it is okay to have other error codes and other fields.

This script expects model names in the LiteLLM format.
It is up to you to ensure that the verifier container supports the specified language,
which is only fed to the prompt.
You can add more directions to the language if needed, e.g., a version number.

The settings above are the recommended settings for Qwen 3 models, based on an experiment with Julia.
If you get many "LM response was truncated" warnings,
you may have forgotten to add "/nothink" to the --language argument to disable thinking.
(A few warnings are inevitable due to repetitive generations.)
