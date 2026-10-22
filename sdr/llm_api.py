import openai 
from tenacity import retry, stop_after_attempt, wait_random_exponential

def gpt_synthetic_prefix(example, prompt, model="gpt-4o-mini"):
    # default generate synthetic prefix from non-member data
    prompt = generate_prompt(example, prompt)

    response = get_completion(prompt, model=model)    
    return response

def generate_prompt(example, prompt):
    return f"{prompt} Don't use any emojis.  Given-text:{example}   Passage:"

@retry(stop=stop_after_attempt(5), wait=wait_random_exponential(min=1, max=10))
def get_completion(prompt, model="gpt-4o-mini", max_tokens=1024):
    message = [{"role": "user", "content": prompt}]
    responses = openai.chat.completions.create(
        model=model,
        messages=message,
        max_tokens=max_tokens,
        temperature=0.5,
    )

    content = responses.choices[0].message.content

    # --- 关键：兜底成字符串 ---
    if isinstance(content, str):
        return content.strip()
    elif isinstance(content, list):
        # 如果是分块，拼接
        texts = [c.get("text", "") for c in content if isinstance(c, dict)]
        return "\n".join(texts).strip()
    elif content is None:
        # 有可能是 function_call/tool_call
        return ""
    else:
        return str(content).strip()