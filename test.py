from openai import OpenAI

client = OpenAI(
  api_key="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"
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "write a haiku about ai"}
  ]
)

print(completion.choices[0].message);
