import requests

problems = [
    "Вычисли: (29 + 50) * 3 - 9",
    "Бригада из 7 человек делает 840 деталей за 4 часов. Сколько деталей сделают 5 человек за 3 часов?",
    "Напиши функцию is_prime(n) на Python.",
]
for p in problems:
    r = requests.post("http://localhost:11434/api/chat", json={
        "model": "izumxai:2b",
        "messages": [{"role": "system", "content": "Ты — IzumxAI, точный и краткий ассистент."},
                     {"role": "user", "content": p}],
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 2048, "num_predict": 384},
    }, timeout=600)
    d = r.json()
    print("Q:", p)
    print("A:", d["message"]["content"][:500])
    print("-" * 70)
