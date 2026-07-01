def paginate(items, page_size, page):
    """Return a page of items. page is 1-indexed."""
    start = page * page_size
    end = start + page_size
    return items[start:end]
