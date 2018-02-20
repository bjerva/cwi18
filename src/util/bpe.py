from math import log


def infer_spaces(s, words):
    """Uses dynamic programming to infer the location of spaces in a string
    without spaces."""
    if not s[0] == " ":
        s = " "+s
    s = s.lower().replace(" ", "▁")
    wordcost = dict((k, log((i+1)*log(len(words)))) for i, k in enumerate(words))
    maxword = max(len(x) for x in words)
    # Find the best match for the i first characters, assuming cost has
    # been built for the i-1 first characters.
    # Returns a pair (match_cost, match_length).

    def best_match(i):
        candidates = enumerate(reversed(cost[max(0, i-maxword):i]))
        return min((c + wordcost.get(s[i-k-1:i], 9e999), k+1)
                   for k, c in candidates)

    # Build the cost array.
    cost = [0]
    for i in range(1, len(s)+1):
        c, k = best_match(i)
        cost.append(c)

    # Backtrack to recover the minimal-cost string.
    out = []
    i = len(s)
    while i > 0:
        c, k = best_match(i)
        assert c == cost[i]
        out.append(s[i-k:i])

        i -= k

    return list((reversed(out)))
