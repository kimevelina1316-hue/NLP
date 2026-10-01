import os
import gensim
import re

# 1. Проверка наличия файла модели
MODEL_PATH = "cbow.txt"
if not os.path.exists(MODEL_PATH):
    print(" Ошибка: Файл 'cbow.txt' не найден в папке со скриптом.")
    exit()

print(" Загрузка векторной модели Word2Vec...")

word2vec = gensim.models.KeyedVectors.load_word2vec_format(MODEL_PATH, binary=False)
print(" Модель успешно загружена.\n")

# 2. Регулярное выражение для фильтрации существительных 
noun_pattern = re.compile(r"(.*)_NOUN")

def to_noun(word):
    """Добавляет суффикс _NOUN, если его нет, для корректного поиска в модели"""
    return word if word.endswith("_NOUN") else f"{word}_NOUN"

def clean_word(word):
    
    match = noun_pattern.match(word)
    return match.group(1) if match else word

def get_top_nouns(model, pos_list, neg_list, topn=20):
    
    pos_nouns = [to_noun(w) for w in pos_list]
    neg_nouns = [to_noun(w) for w in neg_list]

    # Проверяем, есть ли все слова в словаре модели
    for w in pos_nouns + neg_nouns:
        if w not in model:
            return None  # Возвращаем None, если слово не найдено

    # Вычисление линейной комбинации 
    dist = model.most_similar(positive=pos_nouns, negative=neg_nouns, topn=topn)

    # Фильтрация: оставляем ТОЛЬКО имена существительные 
    result = []
    for word, score in dist:
        match = noun_pattern.match(word)
        if match is not None:
            result.append((clean_word(word), score))
        if len(result) == 10:  # Ограничиваем вывод ровно 10 словами
            break
            
    return result


def solve_reverse_task():
    print("-" * 60)
    t1 = input("Введите первое слово (сущ.): ").strip().lower()
    t2 = input("Введите второе слово (сущ.): ").strip().lower()
    print("-" * 60)

    n1, n2 = to_noun(t1), to_noun(t2)
    if n1 not in word2vec or n2 not in word2vec:
        print(" Ошибка: одного из слов нет в словаре модели.")
        return

    print(f" Анализ слов: '{t1}' и '{t2}'")
    
    # Находим соседей для построения умных гипотез
    neighbors1 = [clean_word(w) for w, _ in word2vec.most_similar(n1, topn=10) if noun_pattern.match(w)]
    neighbors2 = [clean_word(w) for w, _ in word2vec.most_similar(n2, topn=10) if noun_pattern.match(w)]

    # Генерируем гипотезы линейных комбинаций
    hypotheses = [
        ([t1, t2], []),  # Простое сложение целевых слов
        ([neighbors1[0], neighbors2[0]], []) if len(neighbors1) > 0 and len(neighbors2) > 0 else None,
        ([neighbors1[0], t2], [t1]) if len(neighbors1) > 0 else None, # Аналогия 1
        ([neighbors2[0], t1], [t2]) if len(neighbors2) > 0 else None, # Аналогия 2
        ([neighbors1[0], neighbors1[1]], []) if len(neighbors1) >= 2 else None,
        ([neighbors2[0], neighbors2[1]], []) if len(neighbors2) >= 2 else None,
    ]
    # Убираем None из списка гипотез
    hypotheses = [h for h in hypotheses if h is not None]

    best_combo = None
    best_hits = 0
    best_result = None

    print(" Проверка вариантов линейных комбинаций...\n")
    for pos, neg in hypotheses:
        res = get_top_nouns(word2vec, pos, neg, topn=20)
        if res is None:
            continue

        top_words = [item[0] for item in res]
        # Считаем, сколько целевых слов попало в Топ-10
        hits = sum(1 for t in [t1, t2] if t in top_words)

        if hits > best_hits:
            best_hits = hits
            best_combo = (pos, neg)
            best_result = res

    # 4. Итоговый вывод для отчёта
    print("=" * 60)
    if best_hits == 2:
        print(" УСПЕХ! Найдена комбинация, где оба слова попадают в Топ-10.")
    else:
        print("Примечание: Слова семантически далеки друг от друга.")

    pos_str = " + ".join(best_combo[0]) if best_combo[0] else "None"
    neg_str = " - ".join(best_combo[1]) if best_combo[1] else "None"

    print(f"\nРЕКОМЕНДУЕМАЯ ЛИНЕЙНАЯ КОМБИНАЦИЯ:")
    print(f"  positive = {best_combo[0]}")
    print(f"  negative = {best_combo[1]}")

    print(f"\nРезультат вычисления (Топ-10 существительных):")
    for i, (w, s) in enumerate(best_result, 1):
        marker = " <-- ЗАДАННОЕ СЛОВО" if w in [t1, t2] else ""
        print(f"  {i}. {w:15} {s:.4f}{marker}")
    print("=" * 60)

# Запуск программы
if __name__ == "__main__":
    print("=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА №2: Векторное представление слов (Word2Vec)")
    print("=" * 60)
    
    solve_reverse_task()