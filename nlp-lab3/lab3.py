from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F
import torch

# Загрузка модели и токенизатора
print("Загружаем модель BERT... (первый раз это займёт 5-10 минут)")
name = 'bert-base-multilingual-uncased'
tokenizer = BertTokenizer.from_pretrained(name)
model = BertForMaskedLM.from_pretrained(name, return_dict=True)
model.eval()
print("Модель загружена!\n")


def predict_mask(sentence):
    """Принимает предложение с [MASK] и выводит топ-10 слов."""
    inputs = tokenizer(sentence, return_tensors="pt")
    mask_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]

    with torch.no_grad():
        output = model(**inputs)

    logits = output.logits[0, mask_index[0], :]
    probs = F.softmax(logits, dim=-1)
    top = torch.topk(probs, 10)

    print(f"\nПредложение: {sentence}")
    print("Топ-10 вариантов для [MASK]:")
    results = []
    for rank, (prob, token_id) in enumerate(zip(top.values, top.indices), 1):
        word = tokenizer.decode([token_id])
        results.append(word)
        print(f"  {rank:2}. {word:<20} (вероятность: {prob.item():.4f})")
    return results


if __name__ == "__main__":
    print("=" * 50)
    print("  BERT Masked Language Model — Топ-10 слов")
    print("=" * 50)
    print("Введите предложение с токеном [MASK]")
    print("Пример: В университете студенты [MASK] целый день.")
    print("Для выхода введите 'exit'\n")

    while True:
        sentence = input("Ваше предложение: ").strip()
        if sentence.lower() == 'exit':
            break
        if '[MASK]' not in sentence and '[mask]' not in sentence:
            print("⚠️  Не забудьте добавить [MASK] в предложение!\n")
            continue
        sentence = sentence.replace('[mask]', '[MASK]')
        predict_mask(sentence)
        print()