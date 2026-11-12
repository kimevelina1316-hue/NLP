import re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, set_seed

MODEL_NAME = "ai-forever/rugpt3large_based_on_gpt2"
TARGET_WORDS = ["индейка", "шампиньоны"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPTS = [
    "Классификация продуктов для праздничного стола.\n"
    "Мясо птицы по размеру:\n"
    "Мелкое — курица.\n"
    "Среднее — утка.\n"
    "Крупное — ",

    "Виды мяса птицы:\n"
    "Самое распространённое — курица.\n"
    "Более крупное и диетическое — ",

    "Продукты для запекания:\n"
    "Мясо птицы — ",

    "Меню ресторана:\n"
    "Блюда из крупной птицы — ",

    "Кулинарный словарь.\n"
    "Крупная домашняя птица с белым мясом — ",
]

TEMPERATURES = [0.7, 0.8, 0.9]
SEEDS = list(range(100, 150))  # 50 seed

def check_prompt(text):
    words = re.findall(r"\w+", text.casefold())
    if any(w.startswith(("индейк", "шампиньон")) for w in words):
        raise ValueError("Промпт содержит запрещённую словоформу!")

def check_result(text):
    words = re.findall(r"\w+", text.casefold())
    first_seen = False
    for word in words:
        if word == TARGET_WORDS[0]:
            first_seen = True
        elif word == TARGET_WORDS[1] and first_seen:
            return True
    return False

def generate(model, tok, prompt, temperature, seed, max_new_tokens=200):
    check_prompt(prompt)
    set_seed(seed)
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    prompt_length = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output = model.generate(
            **inputs, do_sample=True, temperature=temperature,
            top_k=50, top_p=0.95, max_new_tokens=max_new_tokens,
            repetition_penalty=1.0, no_repeat_ngram_size=0,
        )
    return tok.decode(output[0, prompt_length:], skip_special_tokens=True)

def main():
    print(f"Устройство: {DEVICE}")
    print("Загрузка модели...")
    tok = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE)
    model.eval()
    print(" Модель загружена.\n")

    total = len(PROMPTS) * len(TEMPERATURES) * len(SEEDS)
    print(f"Запланировано запусков: {total}\n")

    attempt = 0
    for prompt_idx, prompt in enumerate(PROMPTS, 1):
        for temp in TEMPERATURES:
            for seed in SEEDS:
                attempt += 1
                continuation = generate(model, tok, prompt, temp, seed)
                success = check_result(continuation)
                print(f"[{attempt}/{total}] Промпт #{prompt_idx} | "
                      f"T={temp} | seed={seed} | успех={success}")
                if success:
                    print("\n" + "=" * 60)
                    print(" НАШЛИ УСПЕШНЫЙ ЗАПУСК!")
                    print("=" * 60)
                    print(f"\nПромпт:\n{prompt}")
                    print(f"\nПараметры: temperature={temp}, seed={seed}")
                    print(f"\nПродолжение:\n{continuation}")
                    print("=" * 60)
                    return

    print("\n Ни одна конфигурация не сработала.")

if __name__ == "__main__":
    main()