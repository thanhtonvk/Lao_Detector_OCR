from utils import laos_parser
from collections import defaultdict
import editdistance


def def_value():
    return []


address_hash = defaultdict(def_value)
address_set = set()

with open("data/lao_address_unq.txt", encoding="utf8") as file:
    # for line in file:
    #     arr = laos_parser.parse_text(line.strip().replace("(", "").replace(")", "").replace(" ", "").replace("ຳ", "ໍາ").replace("ໝ", "ຫມ").replace("ໜ", "ຫນ"))
    #     address_set.add("".join(arr))
    #
    #     if "".join(arr) == "ນາຄໍາ":
    #         print(line)
    #
    #     for token in arr:
    #         address_hash[token].append(arr)

    for line in file:
        for c in ['ື', 'ີ', 'ິ', 'ຶ', 'ູ', 'ຸ', 'ົ']:
            if c in line:
                arr = laos_parser.parse_text(line.strip().replace("(", "").replace(")", "").replace(" ", "").replace("ຳ", "ໍາ").replace("ໝ", "ຫມ").replace("ໜ", "ຫນ"))
                address_set.add("".join(arr))


name_set = set()

with open("data/name_uniq.txt", encoding="utf8") as file:
    for line in file:
        for c in ['ື', 'ີ', 'ິ', 'ຶ', 'ູ', 'ຸ', 'ົ']:
            if c in line:
                arr = laos_parser.parse_text(line.strip().replace("(", "").replace(")", "").replace(" ", "").replace("ຳ", "ໍາ").replace("ໝ", "ຫມ").replace("ໜ", "ຫນ"))
                name_set.add("".join(arr))



import re


def lreplace(pattern, sub, string):
    """
    Replaces 'pattern' in 'string' with 'sub' if 'pattern' starts 'string'.
    """
    return re.sub('^%s' % pattern, sub, string)


def rreplace(pattern, sub, string):
    """
    Replaces 'pattern' in 'string' with 'sub' if 'pattern' ends 'string'.
    """
    return re.sub('%s$' % pattern, sub, string)


def post_processing_address(text):
    text = lreplace("ແຂວງ", "", text)
    text = text.strip()
    # text = lreplace("ບ້ານ", "", text)
    # text = text.strip()
    text = lreplace("ຂວງ", "", text)
    text = text.strip()
    text = lreplace("ຂ", "", text)
    text = text.strip()

    # if text in address_set:
    #     return text
    # else:
    #     vowel_set_list = [{'ື', 'ີ', 'ິ', 'ຶ', 'ົ'}, {'ູ', 'ຸ'}]
    #     for idx, c in enumerate(text):
    #         for vowel_set in vowel_set_list:
    #             if c in vowel_set:
    #                 temp_name_part = list(text)
    #                 for candidate_c in vowel_set:
    #                     temp_name_part[idx] = candidate_c
    #
    #                     if "".join(temp_name_part) in address_set:
    #                         print("recorrect_address")
    #                         return "".join(temp_name_part)
    return text

    # # print(address_hash)
    # arr = laos_parser.parse_text(text)
    #
    # if "".join(arr) in address_set:
    #     print("xxx")
    #     return text
    #
    # candidate_set = set()
    # for token in arr:
    #     if token in address_hash:
    #         for candidate in address_hash[token]:
    #             if len(candidate) == len(arr):
    #                 candidate_set.add("".join(candidate))
    #
    # if len(candidate_set) > 0:
    #     rst = ""
    #     min_score = 1000
    #     for candidate in candidate_set:
    #         score = editdistance.eval(candidate, text)
    #         if score < min_score:
    #             rst = candidate
    #             min_score = score
    #
    #     if min_score <= 2:
    #         print("{}: {}".format(rst, min_score))
    #     else:
    #         rst = text
    # else:
    #     rst = text
    #
    # return rst
    return text


print(post_processing_address("ຊອກ"))

def find_nth(s, x, n):
    i = -1
    for _ in range(n):
        i = s.find(x, i + len(x))
        if i == -1:
            break
    return i


def postprocessing_id_number(info):
    id_number_space_pos = find_nth(info["id_number"]["value"], "-", 1)

    if id_number_space_pos == 1:
        id_number = "A" + info["id_number"]["value"]
    else:
        id_number = info["id_number"]["value"]

    return id_number[0:12]


def post_processing_name_part(name_part):
    if name_part in name_set:
        return name_part
    else:
        vowel_set_list = [{'ື', 'ີ', 'ິ', 'ຶ', 'ົ'}, {'ູ', 'ຸ'}]
        for idx, c in enumerate(name_part):
            for vowel_set in vowel_set_list:
                if c in vowel_set:
                    temp_name_part = list(name_part)
                    for candidate_c in vowel_set:
                        temp_name_part[idx] = candidate_c

                        if "".join(temp_name_part) in name_set:
                            print("recorrect_name_part")
                            return "".join(temp_name_part)
    return name_part

# print(post_processing_name_part("ວິຊາປຣາຄາວາມຸຣະ"))



def postprocessing_name(info):
    if "໋" in info["name"]["value"]:
        name = info["name"]["value"].replace("໋", "")
    else:
        name = info["name"]["value"]
    return name


def postprocessing_date(info):
    expiry = info["expiry"]
    issue_date = info["issue_date"]

    expiry_pos = find_nth(expiry["value"], "/", 2)
    issue_date_pos = find_nth(issue_date["value"], "/", 2)

    expiry_date = expiry["value"][:expiry_pos]
    expiry_year = expiry["value"][expiry_pos + 1:]

    issue_date_date = issue_date["value"][:issue_date_pos]
    issue_date_year = issue_date["value"][issue_date_pos + 1:]

    if expiry_date != issue_date_date or int(issue_date_year) + 5 != expiry_year:
        if expiry["conf"] > issue_date["conf"]:
            issue_date_rst = expiry_date + "/" + str(int(expiry_year) - 5)
            expiry_rst = expiry_date + "/" + str(expiry_year)
        else:
            issue_date_rst = issue_date_date + "/" + str(int(issue_date_year))
            expiry_rst = issue_date_date + "/" + str(int(issue_date_year) + 5)
    else:
        expiry_rst = expiry_date + "/" + str(expiry_date)
        issue_date_rst = issue_date_date + "/" + str(int(issue_date_year))

    return issue_date_rst, expiry_rst
