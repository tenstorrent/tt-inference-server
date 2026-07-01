def process(items):
    result = []
    for item in items:
        result.append(item.strip() if isinstance(item, str) else item)
    return result
