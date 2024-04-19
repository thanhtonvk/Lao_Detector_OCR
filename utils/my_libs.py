#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import re

import pandas as pd


def extract_year(dob):
    std_birth_year, std_birth_month, std_birth_day, std_birth_date = parse_dob(str(dob))
    return std_birth_year


def extract_month(dob):
    std_birth_year, std_birth_month, std_birth_day, std_birth_date = parse_dob(str(dob))
    return std_birth_month


def extract_day(dob):
    std_birth_year, std_birth_month, std_birth_day, std_birth_date = parse_dob(str(dob))
    return std_birth_day


def extract_date(dob):
    std_birth_year, std_birth_month, std_birth_day, std_birth_date = parse_dob(str(dob))
    return std_birth_date


def vn_norm_text(text, english=False, low=False, rm_pun=False, rm_space=False):
    import unicodedata
    import unidecode
    if text is None:
        text = ""
    text = str(text)
    if low:
        text = text.lower()
    vietnam_letters = ["a", "á", "à", "ả", "ạ", "ã", "ă", "ắ", "ằ", "ẳ", "ặ", "ẵ", "â", "ấ", "ầ", "ẩ", "ậ", "ẫ", "b",
                       "c", "d", "đ", "e", "é", "è", "ẻ", "ẹ", "ẽ", "ê", "ế", "ề", "ể", "ệ", "ễ", "f", "g", "h", "i",
                       "í", "ì", "ỉ", "ị", "ĩ", "j", "k", "l", "m", "n", "o", "ó", "ò", "ỏ", "ọ", "õ", "ô", "ố", "ồ",
                       "ổ", "ộ", "ỗ", "ơ", "ớ", "ờ", "ở", "ợ", "ỡ", "p", "q", "r", "s", "t", "u", "ú", "ù", "ủ", "ụ",
                       "ũ", "ư", "ứ", "ừ", "ử", "ự", "ữ", "v", "w", "x", "y", "ý", "ỳ", "ỹ", "ỵ", "ỷ", "z", "A", "Á",
                       "À", "Ả", "Ạ", "Ã", "Ă", "Ắ", "Ằ", "Ẳ", "Ặ", "Ẵ", "Â", "Ấ", "Ầ", "Ẩ", "Ậ", "Ẫ", "B", "C",
                       "D", "Đ", "E", "É", "È", "Ẻ", "Ẹ", "Ẽ", "Ê", "Ế", "Ề", "Ể", "Ệ", "Ễ", "F", "G", "H", "I", "Í",
                       "Ì", "Ỉ", "Ị", "Ĩ", "J", "K", "L", "M", "N", "O", "Ó", "Ò", "Ỏ", "Ọ", "Õ", "Ô", "Ố", "Ồ", "Ổ",
                       "Ộ", "Ỗ", "Ơ", "Ớ", "Ờ", "Ở", "Ợ", "Ỡ", "P", "Q", "R", "S", "T", "U", "Ú", "Ù", "Ủ", "Ụ", "Ũ",
                       "Ư", "Ứ", "Ừ", "Ử", "Ự", "Ữ", "V", "W", "X", "Y", "Ý", "Ỳ", "Ỷ", "Ỵ", "Ỹ", "Z", " ", "0", "1",
                       "2", "3", "4", "5", "6", "7", "8", "9", "+", "#", "/", '-', '.', "\n", '&']
    text = unicodedata.normalize('NFC', text)
    text = "".join(i for i in text if i in vietnam_letters).strip()
    if english:
        text = unidecode.unidecode(text)
    if rm_pun:
        text = re.sub(r'/', ' ', text)
        text = re.sub(r'-', ' ', text)
        text = re.sub('\.', ' ', text)
    if rm_space:
        text = text.strip().replace(' ', '')
    text = " ".join(text.split())
    return text


def rm_character_not_num(text):
    return ''.join([i for i in text if i.isdigit()])


def parse_dob(dob):
    std_dob = ""
    std_birth_year = ""
    std_birth_date = ""
    std_birth_month = ""
    std_birth_day = ""
    if len(dob) > 4 and not dob[0:4].isdigit():
        std_dob = dob
        std_birth_year = dob[-4:]
        std_birth_date = dob[:-5]
        std_birth_month = std_birth_date[-2:]
        std_birth_day = std_birth_date[:-2]
    elif len(dob) > 4 and dob[0:4].isdigit():
        std_dob = dob
        std_birth_year = dob[:4]
        std_birth_date = dob[-5:]
        std_birth_month = std_birth_date[:-2]
        std_birth_day = std_birth_date[-2:]
    else:
        std_birth_year = dob
    std_birth_month = rm_character_not_num(std_birth_month)
    std_birth_month = std_birth_month if len(std_birth_month) != 1 else ''.join(['0', std_birth_month])
    std_birth_day = rm_character_not_num(std_birth_day)
    std_birth_day = std_birth_day if len(std_birth_day) != 1 else ''.join(['0', std_birth_day])
    
    return std_birth_year, std_birth_month, std_birth_day, std_birth_date


def norm_dob(dob):
    year, month, day, date = parse_dob(str(dob))
    dob = ""
    if day and month and year:
        dob = '/'.join([day, month, year])
    elif year:
        dob = year
    return dob


def random_name(length=30):
    import random
    import string
    return ''.join(random.choices(string.ascii_uppercase + string.digits + string.ascii_lowercase + "-", k=length))


def is_json(myjson):
    try:
        json_object = json.loads(myjson)
    except ValueError as e:
        print(e)
        return False
    return True
