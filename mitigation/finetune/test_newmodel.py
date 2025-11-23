from vllm import LLM, SamplingParams

llm = LLM(
    model="Tiyamo317/qwen2.5-1.5b-emotional",
    dtype="float16"
)

out = llm.generate(
    ["Hi!"],
    SamplingParams(max_tokens=100)
)

print(out[0].outputs[0].text)
