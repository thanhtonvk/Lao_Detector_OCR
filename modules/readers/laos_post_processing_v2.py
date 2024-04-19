import json
import spellchecker
import os
import difflib

with open("data/address_clean/province.json", "r", encoding="utf-8") as f:
    province_dict = json.load(f)

with open("data/address_clean/district.json", "r", encoding="utf-8") as f:
    district_dict = json.load(f)

with open("data/address_clean/ward.json", "r", encoding="utf-8") as f:
    ward_dict = json.load(f)

with open("data/address_clean/index.json", "r", encoding="utf-8") as f:
    district2idx = json.load(f)

province_spell = spellchecker.SpellChecker(
    local_dictionary="data/address_clean/province.json")
# district_spell = spellchecker.SpellChecker(
#     local_dictionary="data/address_clean/district_v2.json")


def spell_check(name, spell_checker=province_spell):
    if name is None:
        return None
    # province = province.lower()
    suggested_name = spell_checker.correction(name)

    if suggested_name is not None:
        return suggested_name
    else:
        return name


# def spell_check_district(district):
#     return spell_check(district, district_spell)


def spell_check_province(province):
    return spell_check(province, province_spell)


def recorrect(province=None, district=None, ward=None):
    if province is not None:
        # province = spell_check_province(province)
        province_ = difflib.get_close_matches(province, province_dict.keys(), cutoff=0.5)
        if len(province_) > 0:
            province = province_[0]

    if district is not None:
        district_ = difflib.get_close_matches(district, district_dict.keys(), cutoff=0.5)
        if len(district_) > 0:
            district = district_[0]

        if ward is not None:
            if district in district2idx:
                # ward_spell = spellchecker.SpellChecker(local_dictionary=os.path.join(
                #     "data/address_clean/ward", str(district2idx[district]) + ".json"), distance=3)
                # ward = spell_check(ward, ward_spell)
                data = ward_dict[district]
                ward_ = difflib.get_close_matches(ward, data, cutoff=0.5)
                if len(ward_) > 0:
                    ward = ward_[0]
    return province, district, ward

if __name__ == "__main__":
    province = "ຫົວພັນ"
    district = "ຊຳເໜືອ"
    ward = "ຫ້ວຍຫານ"

    p, d, w = recorrect(province, district, ward)

    print(p, d, w)
    print(district2idx[d])
    print(p in province_dict, d in district_dict)
