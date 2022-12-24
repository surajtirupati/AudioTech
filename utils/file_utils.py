from pytube import YouTube
from pydub import AudioSegment
from pydub.utils import which
from typing import Optional
import os
import fitz
import re

from utils.list_utils import find_float_strings_in_list
from utils.string_utils import append_file_extension, remove_special_characters, find_text_between_substrings


def yt_to_audio(link: str, output_path: Optional[str] = None):
    """
    Converts a YouTube link to an audio file and downloads it in the repository.

    Parameters
    ----------
    link: YouTube link to download audio from
    output_path: output path to store file

    Returns
    -------
    audio object, title
    """
    vid_data = YouTube(link)
    vid_audio = vid_data.streams.get_audio_only()
    vid_audio.download(output_path=output_path)
    return vid_audio, vid_data.title


def mp4_to_wav(file_path: str):
    """
    Converts mp4 to wav using pydub AudioSegment
    Parameters
    ----------
    file_path: full path of file to convert

    Returns
    -------
    Nothing - method functionality is implicit; the wav file is rendered to the same location as the original mp4
    """
    AudioSegment.converter = which("ffmpeg")
    dest = append_file_extension(file_path[:-4], ".wav")
    sound = AudioSegment.from_file(file_path, format="mp4")
    sound.export(dest, format="wav")
    return


def convert_yt_link_to_wav_path(link: str, folder: str) -> str:
    """
    Converts a YouTube link to a wav file and returns the path of the wav

    Parameters
    ----------
    link: youtube link
    folder: destination folder of the wav

    Returns
    -------
    string: path of the newly converted wav
    """
    audio_file, title = yt_to_audio(link, folder)
    title = remove_special_characters(title)
    mp4_path = folder + "/{}.mp4".format(title)
    mp4_to_wav(mp4_path)
    os.remove(mp4_path)
    wav_path = append_file_extension(mp4_path[:-4], ".wav")
    return wav_path


def save_txt_file(file_path: str, file_contents: str):
    """
    Saves a text file
    Parameters
    ----------
    file_path: full (relative or absolute) path of the file to save
    file_contents: string contents to save in the file

    Returns
    -------

    """
    with open(file_path, 'w', encoding="utf-8") as f:
        f.write(file_contents)


def open_txt_file(file_path: str):
    """
    Opens a text file
    Parameters
    ----------
    file_path: full (relative or absolute) path of the file to oepn

    Returns
    -------
    Contents of the file
    """
    with open(file_path, 'r', encoding="utf-8") as file:
        data = file.read()

    return data


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from pdf.
    Parameters
    ----------
    pdf_path: path of pdf

    Returns
    -------
    string containing pdf text
    """
    with fitz.open(pdf_path) as doc:
        final_text = ""
        for page in doc:
            final_text += page.get_text()

    return final_text


def get_bold_text(page: fitz.fitz.Page) -> list:
    """
    Extracts bold text from pdf
    Parameters
    ----------
    page: page from PyMuPDF document

    Returns
    -------
    list containing bold text
    """
    bold_text = []
    blocks = page.get_text("dict", flags=11)["blocks"]
    for b in blocks:  # iterate through the text blocks
        for line in b["lines"]:  # iterate through the text lines
            for span in line["spans"]:  # iterate through the text spans
                if span["flags"] == 20:  # 20 targets bold
                    tmp_text = span['text']
                    bold_text.append(tmp_text)

    return bold_text


def extract_titles_from_page(page: fitz.fitz.Page, text: str) -> list:
    """
    Extracts section titles of a research paper. Looks for integer strings in the bold text and stores the integer string
    and the subsequent string (the title of the paper's subsection)
    Parameters
    ----------
    page: page from PyMuPDF document
    text: pdf text

    Returns
    -------
    list containing the titles
    """
    bold_text = get_bold_text(page)
    int_loc = find_float_strings_in_list(bold_text)

    titles = []

    for loc in int_loc:
        regex = r'\b[-:]\b'

        if bold_text[loc + 1] not in text:
            title_word = re.sub(regex, ' \g<0> ', bold_text[loc + 1])  # space between '-'

        else:
            title_word = bold_text[loc + 1]

        if title_word not in text:
            title_word = re.sub(r"(\w)([A-Z])", r"\1 \2", title_word)  # adding spaces between capitalised words

        tmp_title = bold_text[loc] + "\n" + title_word
        titles.append(tmp_title)

    return titles


def extract_titles_from_document(doc: fitz.fitz.Document, text: str) -> list:
    """
    Extracts titles from an entire pdf document
    Parameters
    ----------
    doc: pdf document from PyMuPDF
    text: pdf text

    Returns
    -------
    list of all the titles in the pdf document
    """
    titles = []
    for page in doc:
        page_titles = extract_titles_from_page(page, text)
        titles.extend(page_titles)

    return titles


def get_pdf_dict_of_sections(pdf_path: str) -> dict:
    """
    Returns all the titles and their text for each subsection of the research paper (bar the conclusion)
    Parameters
    ----------
    pdf_path: path of the pdf

    Returns
    -------
    dict containing two keys: title and text; both point to another dict that contains data from each section in numerical
    order
    """
    document = fitz.open(pdf_path)
    text = extract_text_from_pdf('/files/pdfs/Sentence Embeddings using Siamese BERT-Networks.pdf')
    titles = extract_titles_from_document(document, text)

    sections = {"title": {},
                "text": {}}
    for i, title in enumerate(titles):
        if i == len(titles)-1:
            sections["title"][i] = title
            sections["text"][i] = find_text_between_substrings(text, titles[i], "Acknowledgments")

        else:
            sections["title"][i] = title
            sections["text"][i] = find_text_between_substrings(text, titles[i], titles[i + 1])

    return sections
