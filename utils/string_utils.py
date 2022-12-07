import re
from num2words import num2words
from typing import List


def remove_special_characters(title: str):
    special_characters = ['/', '?', '*', '|', ':', '<', '>', '"']
    for special_character in special_characters:
        if special_character in title:
            title = remove_char(title, special_character)

    return title


def remove_char(string: str, char: str):
    # find total no. of
    # occurrence of character
    counts = string.count(char)

    # convert into list
    # of characters
    string = list(string)

    # keep looping until
    # counts become 0
    while counts:
        # remove character
        # from the list
        string.remove(char)

        # decremented by one
        counts -= 1

    # join all remaining characters
    # of the list with empty string
    string = ''.join(string)

    return string


def append_file_extension(string: str, extension: str):
    return string + extension


def count_words(string: str, IN: int = 1, OUT: int = 0):
    state = OUT
    wc = 0

    # Scan all characters one by one
    for i in range(len(string)):

        # If next character is a separator,
        # set the state as OUT
        if (string[i] == ' ' or string[i] == '\n' or
                string[i] == '\t'):
            state = OUT

        # If next character is not a word
        # separator and state is OUT, then
        # set the state as IN and increment
        # word count
        elif state == OUT:
            state = IN
            wc += 1

    # Return the number of words
    return wc


def count_char(string: str, char: str):
    return string.count(char)


def convert_text_to_torch_input(string: str, separator: str = '|'):
    regex = re.compile("[^a-zA-Z0-9']")
    regex = regex.sub(separator, string)
    regex = remove_char(regex, "'")
    regex = [re.sub('(\d+)', lambda m: num2words(m.group()), sentence) for sentence in [regex]][0]
    regex = regex.upper()
    return regex


def convert_sent_list_to_torch_input(sentences: List[str]):
    sent_word_lists = []

    for i, sent in enumerate(sentences):
        temp_sent = convert_text_to_torch_input(sent).split("|")[:-1]
        temp_sent = list(filter(lambda a: a != "", temp_sent))
        sent_word_lists.append(temp_sent)

    return sent_word_lists


def generate_transcript_from_list_of_para(para_list: List[str], bullet_points: bool = False):
    summary = ""

    for para in para_list:

        if not bullet_points:
            summary += f'{para}\n\n'

        else:
            summary += f'{para}\n'

    return summary
