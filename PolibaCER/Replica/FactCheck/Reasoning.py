import os
from openai import OpenAI
from PolibaCER.FactCheck.GoogleAPI import gemini_generate

def get_completion(prompt_template, model, max_tokens=50):
    answer = ""

    if model == "bartowski/Qwen2.5-7B-Instruct-GGUF:Q6_K":
        client = OpenAI(
            base_url="http://127.0.0.1:9931/v1",
            api_key="local"
        )

        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_template}],
            temperature=0.1,
            top_p=0.7,
            max_tokens=max_tokens,
            stream=True
        )

        for chunk in completion:
            if chunk.choices[0].delta.content:
                answer += chunk.choices[0].delta.content

    elif model == "google/diffusiongemma-26b-a4b-it":
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ.get("NVIDIA_API_KEY")
        )

        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_template}],
            temperature=0.1,
            top_p=0.7,
            max_tokens=max_tokens,
            stream=True
        )

        for chunk in completion:
            if chunk.choices[0].delta.content:
                answer += chunk.choices[0].delta.content
    else:
        answer = gemini_generate(prompt_template, model=model)

    return answer