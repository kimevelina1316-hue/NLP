import re
from nltk.tokenize import sent_tokenize, word_tokenize
import pymorphy3

def get_features(tag):
    """Извлекает род, число и падеж из тега pymorphy3"""
    features = {}
    if 'masc' in tag:
        features['gender'] = 'masc'
    elif 'femn' in tag:
        features['gender'] = 'femn'
    elif 'neut' in tag:
        features['gender'] = 'neut'

    if 'sing' in tag:
        features['number'] = 'sing'
    elif 'plur' in tag:
        features['number'] = 'plur'

    for case in ['nomn', 'gent', 'datv', 'accs', 'ablt', 'loct']:
        if case in tag:
            features['case'] = case
            break
    return features

def main():
    filename = "text.txt"

    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    sentences = sent_tokenize(text)
    tokens = []
    for sentence in sentences:
        tokens.extend(word_tokenize(sentence))

    tokens = [token for token in tokens if re.match(r'^[а-яА-ЯёЁ]+$', token)]

    morph = pymorphy3.MorphAnalyzer()
    found_pairs = set()

    for i in range(len(tokens) - 1):
        word1 = tokens[i].lower()
        word2 = tokens[i + 1].lower()

        parse1 = morph.parse(word1)[0]
        parse2 = morph.parse(word2)[0]

        tag1 = str(parse1.tag)
        tag2 = str(parse2.tag)

        is_noun_adj1 = 'NOUN' in tag1 or 'ADJ' in tag1 or 'ADJS' in tag1
        is_noun_adj2 = 'NOUN' in tag2 or 'ADJ' in tag2 or 'ADJS' in tag2

        if not (is_noun_adj1 and is_noun_adj2):
            continue

        feat1 = get_features(tag1)
        feat2 = get_features(tag2)

        match = True
        for key in ['gender', 'number', 'case']:
            if key in feat1 and key in feat2:
                if feat1[key] != feat2[key]:
                    match = False
                    break

        if match:
            found_pairs.add((parse1.normal_form, parse2.normal_form))

    print(f"Найдено пар: {len(found_pairs)}")
    print("-" * 40)
    for p1, p2 in sorted(found_pairs):
        print(f"{p1} {p2}")

if __name__ == "__main__":
    main()