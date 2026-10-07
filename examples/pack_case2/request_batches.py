"""Judge a streamed shortlist in 16-item search order, including the final tail."""


def judge_in_order(judge, checks, entries, shared, refusals):
    items, places = [], []

    def send():
        yield from judge.iter_check_every(
            checks, items, shared, places=places, refusals=refusals, keep_order=True
        )

    for item, place in entries:
        items.append(item)
        places.append(place)
        if len(items) == 16:
            yield from send()
            items.clear()
            places.clear()
    if items:
        yield from send()
