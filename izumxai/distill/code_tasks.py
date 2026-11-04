"""Algorithmic tasks with randomized checks against independent small-input oracles."""
from bisect import bisect_left


def lower_bound_cases(rng):
    cases = [([], 3, 0), ([1, 2, 2, 2, 7], 2, 1), ([1, 2], 9, 2)]
    for _ in range(25):
        a = sorted(rng.randint(-8, 8) for _ in range(rng.randint(0, 30)))
        x = rng.randint(-10, 10)
        cases.append((a, x, bisect_left(a, x)))
    return cases


def range_sums_cases(rng):
    cases = [([], [[0, 0]], [0]), ([2, -3, 5], [[0, 3], [1, 1], [1, 2]], [4, 0, -3])]
    for _ in range(25):
        a = [rng.randint(-20, 20) for _ in range(rng.randint(0, 20))]
        queries = []
        for _ in range(rng.randint(0, 12)):
            left = rng.randint(0, len(a))
            queries.append([left, rng.randint(left, len(a))])
        cases.append((a, queries, [sum(a[left:right]) for left, right in queries]))
    return cases


def longest_unique_cases(rng):
    cases = [("", 0), ("abba", 2), ("abcabcbb", 3), ("АаА", 2)]
    for _ in range(25):
        s = "".join(rng.choice("abcdeАБ ") for _ in range(rng.randint(0, 20)))
        best = max((end - start for start in range(len(s) + 1)
                    for end in range(start, len(s) + 1)
                    if len(set(s[start:end])) == end - start), default=0)
        cases.append((s, best))
    return cases


def min_coins_cases(rng):
    cases = [([], 0, 0), ([], 3, -1), ([1, 3, 4], 6, 2), ([2], 3, -1)]
    for _ in range(25):
        coins = [rng.randint(1, 10) for _ in range(rng.randint(0, 5))]
        amount = rng.randint(0, 30)
        # Breadth-first exploration, independent of the DP reference.
        frontier, seen, steps, answer = {0}, {0}, 0, -1
        while frontier:
            if amount in frontier:
                answer = steps
                break
            frontier = {value + coin for value in frontier for coin in coins
                        if value + coin <= amount and value + coin not in seen}
            seen.update(frontier)
            steps += 1
        cases.append((coins, amount, answer))
    return cases


def shortest_path_cases(rng):
    cases = [(1, [], 0, 0, 0), (3, [[0, 1], [1, 2]], 0, 2, 2),
             (3, [[0, 1], [1, 2]], 2, 0, -1), (2, [], 0, 1, -1)]
    for _ in range(25):
        n = rng.randint(1, 8)
        edges = [[a, b] for a in range(n) for b in range(n) if rng.random() < 0.18]
        start, end = rng.randrange(n), rng.randrange(n)
        # Floyd-Warshall oracle, independent of the BFS reference.
        dist = [[0 if a == b else n + 1 for b in range(n)] for a in range(n)]
        for a, b in edges:
            dist[a][b] = min(dist[a][b], 1)
        for k in range(n):
            for a in range(n):
                for b in range(n):
                    dist[a][b] = min(dist[a][b], dist[a][k] + dist[k][b])
        result = dist[start][end]
        cases.append((n, edges, start, end, result if result <= n else -1))
    return cases


ALGORITHM_TASKS = [
    ("lower_bound", """def lower_bound(a, x):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo
""", "Напиши функцию lower_bound(a, x). Список целых a отсортирован по возрастанию, "
     "возможны повторы. Верни индекс первого элемента >= x или len(a), если его нет. "
     "Для пустого списка верни 0. Используй двоичный поиск.", lower_bound_cases, "hard"),
    ("range_sums", """def range_sums(a, queries):
    prefix = [0]
    for x in a:
        prefix.append(prefix[-1] + x)
    return [prefix[right] - prefix[left] for left, right in queries]
""", "Напиши функцию range_sums(a, queries). a — список целых; каждый запрос [left, right] "
     "задаёт полуинтервал a[left:right], правая граница не включается. Границы корректны: "
     "0 <= left <= right <= len(a). Верни список сумм, используя префиксные суммы. "
     "Пустой полуинтервал имеет сумму 0.", range_sums_cases, "hard"),
    ("longest_unique", """def longest_unique(s):
    last, left, best = {}, 0, 0
    for right, ch in enumerate(s):
        left = max(left, last.get(ch, -1) + 1)
        last[ch] = right
        best = max(best, right - left + 1)
    return best
""", "Напиши функцию longest_unique(s), возвращающую длину самой длинной непрерывной "
     "подстроки без повторяющихся символов. Регистр учитывается, пробел — обычный символ. "
     "Для пустой строки верни 0. Используй скользящее окно.", longest_unique_cases, "hard"),
    ("min_coins", """def min_coins(coins, amount):
    dp = [0] + [amount + 1] * amount
    for value in range(1, amount + 1):
        for coin in coins:
            if coin <= value:
                dp[value] = min(dp[value], dp[value - coin] + 1)
    return dp[amount] if dp[amount] <= amount else -1
""", "Напиши функцию min_coins(coins, amount). Номиналы coins — положительные целые, "
     "количество монет каждого номинала не ограничено. amount — неотрицательное целое. "
     "Верни минимальное число монет для точной суммы или -1, если собрать её невозможно. "
     "Для amount=0 верни 0; coins может быть пустым. Используй динамическое программирование.",
     min_coins_cases, "hard"),
    ("shortest_path", """def shortest_path(n, edges, start, end):
    from collections import deque
    graph = [[] for _ in range(n)]
    for a, b in edges:
        graph[a].append(b)
    distance = [-1] * n
    distance[start] = 0
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node == end:
            return distance[node]
        for other in graph[node]:
            if distance[other] == -1:
                distance[other] = distance[node] + 1
                queue.append(other)
    return -1
""", "Напиши функцию shortest_path(n, edges, start, end) для ориентированного невзвешенного "
     "графа с вершинами от 0 до n-1, n >= 1. Каждая пара [a, b] задаёт ребро только из a в b. "
     "Верни минимальное число рёбер от start до end или -1, если пути нет. start и end "
     "корректны; путь из вершины в себя имеет длину 0. Возможны циклы и петли. Используй BFS.",
     shortest_path_cases, "hard"),
]
