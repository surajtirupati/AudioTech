def subfinder(sub_list, original_list):
    found = False
    sub_list_len = len(sub_list)

    for ind in (idx for idx, e in enumerate(original_list) if e == sub_list[0]):
        if original_list[ind:ind + sub_list_len] == sub_list:
            start, end = ind, (ind + sub_list_len - 1)
            found = True

    if found:
        return start, end

    else:
        return None, None


def subfinder_bool(sub_list, original_list):
    found = False
    sub_list_len = len(sub_list)

    for ind in (idx for idx, e in enumerate(original_list) if e == sub_list[0]):
        if original_list[ind:ind + sub_list_len] == sub_list:
            start, end = ind, (ind + sub_list_len - 1)
            found = True

    if found:
        return found

    else:
        return found


def find_float_strings_in_list(a_list: list) -> list:
    def isfloat(num):
        try:
            float(num)
            return True

        except ValueError:
            return False

    return [i for i, s in enumerate(a_list) if isfloat(s)]
