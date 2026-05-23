import os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("LONGEVITY_BASE_URL", "https://swchnq0ekc3scmqw.us-east-2.aws.endpoints.huggingface.cloud/v1"),
    api_key=os.environ.get("HF_TOKEN", ""),
)

# Test 1 — does it follow JSON format?
r = client.chat.completions.create(
    model="longevity-llm",
    messages=[{"role": "user", "content": "A 67 year old male, BMI 25, good health. How long will he live? Return JSON only: {\"prediction_months\": <number>, \"reasoning\": \"<explanation>\"}"}],
    max_tokens=300,
    temperature=0.0
)
print("=== JSON TEST ===")
print(r.choices[0].message.content)

# Test 2 — does thinking mode expose the think block?
r2 = client.chat.completions.create(
    model="longevity-llm",
    messages=[{"role": "user", "content": "A 67 year old male, BMI 25, good health. How long will he live?"}],
    max_tokens=1000,
    temperature=0.0,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}}
)
print("\n=== THINKING TEST ===")
print(r2.choices[0].message.content)

if hasattr(r2.choices[0].message, 'reasoning_content'):
    print("\n=== REASONING CONTENT ===")
    print(r2.choices[0].message.reasoning_content)
