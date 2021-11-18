import string

allow_list = string.ascii_uppercase + string.digits


def process_allowlist(block_list):
    all_list = allow_list
    if block_list is not None:
        for char in block_list:
            all_list = allow_list.replace(char, "")
    return all_list
