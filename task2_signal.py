import re
import torch

# Define prompts with customizable signal strength and threshold
SENTIMENT_PROMPT = """Task: Analyze the following news headline about a stock and provide a sentiment score between -{signal_strengh} and {signal_strengh}, where:
- -{signal_strengh} means very negative sentiment
- -{threshold} means neutral negative sentiment
- 0 means neutral sentiment
- {threshold} indicates neutral positive sentiment
- {signal_strengh} means very positive sentiment

Do not provide any explanations. Output only a single number in the range of -{signal_strengh} to {signal_strengh} based on the sentiment of the news.

News headline: "{news}"

Generate only a single integer value for the sentiment score after the colon. Sentiment score:
"""

VOLATILITY_PROMPT = """Task: Analyze the following news and recent stock price data to predict the volatility of the stock over the next few days. Provide a volatility score between 0 and {signal_strengh}, where:
- 0 means very low volatility
- {threshold} means moderate volatility
- {signal_strengh} means very high volatility

Do not provide any explanations. Output only a single number in the range of 0 to {signal_strengh} based on the expected volatility.

News: "{news}"

Recent Stock Prices: "{prices}"

Generate only a single integer value for the volatility score after the colon. Volatility score:
"""

def _generate_signal(tokenizer, model, device, news, signal_strengh=10, threshold=3):
    prompt = SENTIMENT_PROMPT.format(signal_strengh=signal_strengh, threshold=threshold, news=news)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    generated_ids = inputs["input_ids"]
    log_probs = []
    max_new_tokens = 5

    for _ in range(max_new_tokens):
        outputs = model(input_ids=generated_ids)
        logits = outputs.logits

        next_token_logits = logits[:, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)

        next_token_id = torch.multinomial(next_token_probs, num_samples=1)

        token_log_prob = torch.log(next_token_probs[0, next_token_id[0, 0]])
        log_probs.append(token_log_prob)

        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

    total_log_prob = torch.stack(log_probs).sum()

    output_string = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    match = re.search(r"Sentiment score:\s*(-?\d+(?:\.\d+)?)", output_string)
    sentiment_score = float(match.group(1)) if match else 0

    return sentiment_score, total_log_prob

def _generate_volatility_signal(tokenizer, model, device, news, prices, signal_strengh=10, threshold=5):
    prompt = VOLATILITY_PROMPT.format(signal_strengh=signal_strengh, threshold=threshold, news=news, prices=prices)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    generated_ids = inputs["input_ids"]
    log_probs = []
    max_new_tokens = 5

    for _ in range(max_new_tokens):
        outputs = model(input_ids=generated_ids)
        logits = outputs.logits

        next_token_logits = logits[:, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)

        next_token_id = torch.multinomial(next_token_probs, num_samples=1)

        token_log_prob = torch.log(next_token_probs[0, next_token_id[0, 0]])
        log_probs.append(token_log_prob)

        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

    total_log_prob = torch.stack(log_probs).sum()

    output_string = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    match = re.search(r"Volatility score:\s*(\d+(?:\.\d+)?)", output_string)
    volatility_score = float(match.group(1)) if match else 0

    return volatility_score, total_log_prob

def generate_signal(tokenizer, model, device, news, signal_strengh=10, threshold=3):
    return _generate_signal(tokenizer, model, device, news, signal_strengh, threshold)

def generate_volatility_signal(tokenizer, model, device, news, prices, signal_strengh=10, threshold=5):
    return _generate_volatility_signal(tokenizer, model, device, news, prices, signal_strengh, threshold)
