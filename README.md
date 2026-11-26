# GSQ-RCO model evaluation

We use LightEval for GPQA-Diamond and AIME25, and the LiveCodeBench source runner for code generation on the `v6` subset. All three benchmarks send requests to a local OpenAI-compatible server. The examples use `inkling-2-mini` as the model name.

## 1. Serve the model

**vLLM.** For faster evaluation, we usually dequantize the GGUF weights and save them as a Hugging Face BF16 checkpoint with the model's tokenizer and chat template. Use a converter that supports Inkling 2 Mini and a vLLM build with Inkling and DSpark support. Keep the matching `mtp.safetensors` drafter and the `mtp_config` metadata in the Hugging Face checkpoint directory. Set the API model name with `--served-model-name`. [vLLM Inkling 2 Mini recipe](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini) · [Inkling 2 Mini checkpoint files](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/main)

```bash
vllm serve /path/to/inkling-2-mini-bf16 \
  --port 8000 \
  --served-model-name inkling-2-mini \
  --reasoning-parser inkling \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
  --default-chat-template-kwargs '{"thinking":true,"reasoning_effort":"xhigh"}' \
  --enable-auto-tool-choice --tool-call-parser inkling
```

**llama.cpp.** Use a build with Inkling 2 Mini and DSpark support, the target GGUF with its Inkling chat template, and a matching DSpark draft GGUF. Set the API model name with `--alias`. With `--jinja`, llama.cpp selects the Inkling parser for reasoning and tool calls from the model's chat template. [llama.cpp server reference](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) · [DSpark setup and draft conversion](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md#dspark-draft-dspark) · [Inkling chat parser](https://github.com/ggml-org/llama.cpp/pull/32381)

```bash
llama-server --model /path/to/inkling-2-mini.gguf \
  --model-draft /path/to/inkling-2-mini-dspark.gguf \
  --host 127.0.0.1 --port 8080 \
  --alias inkling-2-mini --ctx-size 256000 \
  --parallel 10 --kv-unified \
  --flash-attn on --jinja \
  --reasoning on --reasoning-effort xhigh \
  --spec-type draft-dspark --spec-draft-n-max 7 \
  --spec-draft-sampling probabilistic \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0
```

The evaluation clients request up to 128,000 output tokens per completion. With llama.cpp, `--parallel 10` allows the ten completions in an AIME25 request, and `--kv-unified` shares the 256,000-token context pool across the slots. The pool must fit the active prompts and generations. Before running a benchmark, check that `/v1/models` lists `inkling-2-mini` on the port you chose. [llama.cpp completion limits](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-schema.cpp)

## 2. Run GPQA-Diamond and AIME25 with LightEval

Install LightEval with its LiteLLM extra in a Python environment. [LightEval installation](https://huggingface.co/docs/lighteval/en/installation)

```bash
python -m pip install 'lighteval[litellm]'
```

For GPQA-Diamond, accept the dataset's access terms on Hugging Face and authenticate with `hf auth login` in the evaluation environment. [GPQA dataset](https://huggingface.co/datasets/Idavidrein/gpqa) · [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)

For vLLM, save the following as `inkling_vllm_litellm.yaml`. The part after `hosted_vllm/` must match `--served-model-name`. [LightEval LiteLLM setup](https://huggingface.co/docs/lighteval/en/use-litellm-as-backend)

```yaml
model_parameters:
  provider: "hosted_vllm"
  model_name: "hosted_vllm/inkling-2-mini"
  base_url: "http://localhost:8000/v1"
  api_key: ""
  timeout: 10000
  concurrent_requests: 100
  max_model_length: 256000
  generation_parameters:
    temperature: 1.0
    top_p: 1.0
    min_p: 0.0
    max_new_tokens: 128000
```

For llama.cpp, save the following as `inkling_llama_litellm.yaml`. The part after `openai/` must match `--alias`. If the OpenAI client requires a key for the unauthenticated local server, set `api_key` to a placeholder such as `local`.

```yaml
model_parameters:
  provider: "openai"
  model_name: "openai/inkling-2-mini"
  base_url: "http://localhost:8080/v1"
  api_key: ""
  timeout: 10000
  concurrent_requests: 1
  max_model_length: 256000
  generation_parameters:
    temperature: 1.0
    top_p: 1.0
    min_p: 0.0
    max_new_tokens: 128000
```

Run GPQA-Diamond with three samples per question and report mean accuracy (`pass@1`).

```bash
lighteval endpoint litellm inkling_vllm_litellm.yaml "gpqa:diamond@n=3|0" --output-dir results
```

Run AIME25 with ten samples per problem and report average accuracy (`avg@10`).

```bash
lighteval endpoint litellm inkling_vllm_litellm.yaml "aime25@n=10|0" --output-dir results
```

In both commands, `|0` selects zero-shot evaluation. Use `inkling_llama_litellm.yaml` in either command to run against llama.cpp. The server must support returning the requested number of completions per chat request. [GPQA task definition](https://github.com/huggingface/lighteval/blob/main/src/lighteval/tasks/tasks/gpqa.py) · [AIME task definition](https://github.com/huggingface/lighteval/blob/main/src/lighteval/tasks/tasks/aime.py)

## 3. Run LiveCodeBench v6

Clone LiveCodeBench and install it in its own Python environment. The `datasets` constraint keeps support for the dataset loader used by the runner. [LiveCodeBench installation](https://github.com/LiveCodeBench/LiveCodeBench#installation) · [Dataset loader compatibility](https://github.com/LiveCodeBench/LiveCodeBench/issues/107)

```bash
git clone https://github.com/LiveCodeBench/LiveCodeBench.git
cd LiveCodeBench
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e . 'datasets>=3.2,<4'
```

In `lcb_runner/lm_styles.py`, add this entry inside `LanguageModelList`. [LiveCodeBench model registry](https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/lm_styles.py)

```python
    LanguageModel(
        "inkling-2-mini",  # Must match --served-model-name or --alias
        "inkling-2-mini",  # Name used for output paths
        LMStyle.OpenAIChat,
        datetime(2026, 1, 1),
        link=None,
    ),
```

Point the runner at the local server. Use port `8080` for llama.cpp. `local` is a placeholder for a server with authentication disabled. [LiveCodeBench API runner](https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/runner/oai_runner.py)

```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_KEY="local"
```

From the LiveCodeBench checkout, run code generation and grade the resulting solutions.

```bash
python -m lcb_runner.runner.main \
  --model inkling-2-mini \
  --scenario codegeneration \
  --release_version v6 \
  --evaluate \
  --n 3 \
  --temperature 1.0 \
  --top_p 1.0 \
  --max_tokens 128000 \
  --multiprocess 100 \
  --openai_timeout 100000 \
  --use_cache \
  --cache_batch_size 175
```

`--n 3` requests three solutions per problem. `--multiprocess 100` uses up to 100 workers for concurrent generation requests. Report `pass@1` from the `*_eval.json` file under `output/inkling-2-mini/`. The `v6` tag selects the problems added in version 6. The `release_v6` tag includes all problems through version 6. [Runner arguments](https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/runner/parser.py) · [Dataset version definitions](https://huggingface.co/datasets/livecodebench/code_generation_lite/blob/main/code_generation_lite.py)

If the runner fails to import `HUMAN_PROMPT` or `AI_PROMPT`, run `uv pip install 'anthropic==0.42.0'` in the LiveCodeBench environment and retry. [LiveCodeBench import issue](https://github.com/LiveCodeBench/LiveCodeBench/issues/163) · [Anthropic SDK v0.42.0 exports](https://github.com/anthropics/anthropic-sdk-python/blob/v0.42.0/src/anthropic/__init__.py)
