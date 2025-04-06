from dotenv import load_dotenv
import os
load_dotenv()

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

def chat_with_gpt4o(prompt, model="gpt-4o", system_prompt=""):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

def chat_with_deepseek(prompt, model="deepseek-r1", system_prompt=""):
    """
    Call the DeepSeek reasoning model to generate responses.
    
    Args:
        prompt (str): The user prompt to send to the model
        model (str): The model name, defaults to "deepseek-r1"
        system_prompt (str): Optional system prompt to guide the model's behavior
        
    Returns:
        str: The model's response text
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        # Fallback to GPT-4o if DeepSeek is not available
        print(f"Error calling DeepSeek model: {e}. Falling back to GPT-4o.")
        return chat_with_gpt4o(prompt, model="gpt-4o", system_prompt=system_prompt)

def get_embedding(text, model="text-embedding-3-large"):
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding
