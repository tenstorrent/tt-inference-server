def validate(token):
    if not token:
        return False
    if len(token) < 8:
        return False
    if not token.isalnum():
        return False
