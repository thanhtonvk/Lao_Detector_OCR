import spellchecker

province_spell = spellchecker.SpellChecker(local_dictionary="data/id_province.json")
district_spell = spellchecker.SpellChecker(local_dictionary="data/household_district.json")
ward_spell = spellchecker.SpellChecker(local_dictionary="data/household_ward.json")

month_spell = spellchecker.SpellChecker(local_dictionary="data/month.json")



def spell_check(name, spell_checker=province_spell):
    if name is None:
        return None
    # province = province.lower()
    suggested_name = spell_checker.correction(name)

    if suggested_name is not None:
        return suggested_name
    else:
        return name
    

def spell_check_district(district):
    return spell_check(district, district_spell)

def spell_check_province(province):
    return spell_check(province, province_spell)

def spell_check_ward(ward):
    return spell_check(ward, ward_spell)

def spell_check_month(month):
    return spell_check(month, month_spell)