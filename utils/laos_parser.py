# -*- coding: utf-8 -*-

consonants = ["ກ", "ຂ", "ຄ", "ງ", "ຈ", "ສ", "ຊ", "ຍ", "ດ", "ຕ", "ຖ", "ທ", "ນ", "ບ", "ປ", "ຜ", "ຝ", "ພ", "ຟ", "ມ", "ຢ",
              "ລ", "ວ", "ຫ", "ອ", "ຮ", "ຣ", "ໜ", "ໝ", "ເ"]
vowels = ["ະ", "າ", "ແ", "ໂ", "ວ", "ໃ", "ຍ", "ໄ", "ຽ", "ໆ", "ล", "ຊ", "ຯ"]

numbers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]

special = [".", " ", ",", ":", "-", "/", "A", "+", "|", "+", ">", "<"]

consonants.extend(vowels)
main_character = sorted(list(set(consonants)))

numbers.extend(special)
other_character = sorted(list(set(numbers)))

vowels_sub = ['ໍ', 'ິ', 'ີ', 'ຶ', 'ື', 'ັ', 'ົ', '໌', 'ູ', 'ຼ', 'ຸ', '່', '໋', '້', '໊']

top_vowel_sub = ['ື', 'ໍ', 'ັ', 'ີ', 'ົ', 'ິ', 'ຶ', '໌']

rule_top = [("ໍ", "ື"), ("ໍ", "ັ"), ("ໍ", "ີ"), ("ໍ", "ົ"), ("ໍ", "ິ"), ("ໍ", "ຶ"), ("ໍ", "໌"),
            ("ິ", "ີ"), ("ິ", "ຶ"), ("ິ", "ື"),
            ("ຶ", "ື"), ("ີ", "ື")
            ]
invalid_rule = [("ື", "ົ"), ("ີ", "ົ"), ("ິ", "ົ"), ("ຶ", "ົ"),
                ("ື", "໌"), ("ີ", "໌"), ("ິ", "໌"), ("ຶ", "໌"),
                ("ື", "ັ"), ("ີ", "ັ"), ("ິ", "ັ"), ("ຶ", "ັ"),
                ("ີ", "ຶ"),
                ("ັ", "ົ"),
                ("ົ", "໌"),
                ("ູ", "ຼ"),
                ("ູ", "ຸ"),
                ("ຸ", "ຼ")]
low_vowel_sub = ['ຸ', 'ູ', 'ຼ']

compounds = set(
    ['ກ໋ໍ', 'ຜີຸ', 'ຊື໋', 'ວຸ', 'ຮຼ', 'ວື້', 'ມືູ', 'ນູ', 'ຄ່', 'ຄຸ', 'ວິ໊', 'ພີ', 'ຢູ', 'ຂັ່', 'ທັ', 'ຜັູ້', 'ບ່ໍ',
     'ຂື', 'ດີ໊', 'ລູ໊', 'ມີຸ', 'ຖ໋', 'ບົຼ່', 'ຊ້ໍ', 'ຄູົ', 'ນີູ', 'ຈັຸ', 'ຫ່້', 'ອຸ໋', 'ອ້ໍ', 'ຂຶູ', 'ຫິຸ້', 'ຢິ',
     'ຍຸໍ', 'ຜົ່', 'ຈ້ໍ', 'ຂຶ່', 'ຝ້', 'ຊົ່້', 'ຮ່ໍ', 'ທົ້', 'ກຶ້', 'ຕູ້', 'ຄຸ່', 'າຸ', 'ດັ່', 'ທ້ໍ', 'ພືຼ່', 'ຮື່້',
     'ລູ່ໍ', 'ລື່', 'ວັຸ້', 'ຈືູ', 'ປູ້', 'ຜົ້', 'ຂູົ', 'ກີູ', 'ກູົ່', 'ຢ໋ໍ', 'ຮ້', 'ຈັຼ', 'ເົ່', 'ສັຼ', 'ພືຼ', 'ລືູ່',
     'ງືູ່', 'ກູ່ໍ', 'ເົ', 'ຄັຼ', 'ກ້', 'ຝ່ໍ', 'ແູ', 'ຢັ່', 'ຕົ່', 'ຊື່້', 'ໂັ', 'ຢົ', 'ຊົ', 'ດຸ່', 'ຜັູ', 'ຊີ', 'ຖິ',
     'ບັຼ້', 'ຜີຸ້', 'ຄຸ໊', 'ມຶ໊', 'ລ໌', 'ຄິູ່', 'ງ້', 'ຕິຼ', 'ນ໌', 'ອຼ່', 'ຕັ່', 'ປ໋ໍ', 'ດັ້', 'ຄໍ', 'ວ໋', 'ພື່',
     'ວົ້', 'ຂັຸ', 'ລັ້', 'ຢຶ໋', 'ຊ່້', 'ອືູ່', 'ຕຸ໋', 'ພັ໊', 'ຊູົ່', 'ດັູ', 'ຽີ່', 'ປີຸ້', 'ຂ່້', 'ອ່ໍ', 'ຍີຼ', 'າື້',
     'ປັຼ', 'ສືູ່', 'ອິ່', 'ຢື້', 'ມູ໋', 'ສີູ່', 'ຍືຼ', 'ວີ໋', 'ກຼ', 'ດົ້', 'ຄືູ', 'ຟຶ້', 'ເ້', 'ຫືຼ', 'ຕົຼ', 'ວຸ່',
     'ນຶ່້', 'ດີ', 'ລີຸ້', 'ປິ່', 'ງົ່', 'ຝໍ', 'ລໍ', 'ຮັ໋', 'ໃໍ', 'ຣ່ໍ', 'ນິ໊', 'ໄ໋', 'ຕູ່', 'ຊີ່', 'ຣຸ້', 'ຈຸ໊', 'ຄ່້',
     'ຫູ', 'ຄັູ', 'ອົ້', 'ກ້ໍ', 'ດີູ່', 'ຄັ້', 'ອຸ້', 'ຊົ້', 'ສັູ', 'ທີ໋', 'ຜັ່', 'ຍ໊', 'ນັ໊', 'ແຸ', 'ຂ໊ໍ', 'ຖື້',
     'ທຸ໊', 'ນູ່', 'ຊືຼ', 'ນີ່້', 'ທ້', 'ຂ່', 'ບັ້', 'ກີ້', 'ກີຸ່', 'ຈ໊', 'ຫີ່', 'ຄີ້', 'ຟັ໋', 'ຮຸົ້', 'ສ໌', 'ພຶ່',
     'ບຸ້', 'ຍູົ້', 'ຫື', 'ມັ່', 'ຖັ', 'ນື່', 'ຮູ່', 'ຊືຼ່', 'ຖຼ', 'ສິ້', 'ຂູ່', 'ຊັຸ', 'ມົ໊', 'ເ່ໍ', 'ຢຸ້', 'ງິ້',
     'ແໍ', 'ເໍ', 'ຫິ໋', 'ຝື', 'ບຶ່', 'ແ໌', 'ກີ່', 'ຕື່', 'ນຼ້ໍ', 'ວຸ້', 'ໃູ່', 'ຢຼ່', 'ຈົ໋', 'ຕີ່', 'ຕົ່້', 'ລູ້',
     'ຫຼ່', 'ໄື', 'ບືຼ', 'ງໍ', 'ຄູ່', 'ຄ໋', 'ທົ່', 'ຂັຼ', 'ອ່', 'ຝ່', 'ຂຸ້', 'ຜີ້', 'ຖູົ', 'ວັ', 'ນຼໍ', 'ພໍ', 'ຖືຼ',
     'ຕັຼ້', 'ຕ່', 'ຊຸົ', 'ຮູົ້', 'ຫັຼ', 'ຈຼ', 'ມູ້', 'ຫຼ່ໍ', 'ປ່', 'ຝັຼ່', 'ຕັ', 'ກີຸ', 'ທ່້', 'າູ', 'ສ້ໍ', 'ພີູ',
     'ນົ໊', 'ຮິ່', 'ປຼ', 'ຜິ່', 'ມຸ້', 'ປົ໋', 'ຮີ໋', 'ອິ໊', 'ຟືຸ້', 'ຍ໋', 'ດີູ', 'ທ໊ໍ', 'ຮຸ໊', 'ສຼ່', 'ຈົ່', 'ທ໋',
     'ຕີູ້', 'ມືຼ່', 'ສົ່', 'ອຶູ້', 'າຼ', 'ບີ່', 'ຟຼ໌', 'ກື', 'ພືຸ', 'ບຼໍ', 'ກົ', 'ມ່້', 'ດິ໋', 'ຜືູ້', 'ອັ໋', 'ຈັ໊',
     'ດຼ', 'ບື', 'ຟື້', 'ລູ', 'ງີ', 'ນື່້', 'ນືຸ້', 'ຍັູ້', 'ມ້', 'ຖັ້', 'ດຸ໊', 'ຕິ້', 'ຜື້', 'ຕັ້', 'ຣັ້', 'ຂິ້',
     'ຈູ໋', 'ສີ', 'ຍ້', 'ຟ້', 'ບັຼ', 'ປຶ', 'ຕີ໊', 'ຍັູ', 'ດັູ່', 'ລົ່', 'ນ່້ໍ', 'ຊ໊', 'ຜ້', 'ລູ່', 'ຟົ່', 'ຮັູ້', 'ມຸ່',
     'ບ໊', 'ຈີ໊', 'ກິຼ', 'ງີ້', 'ຟິຼ', 'ກ໌', 'ງ໋', 'ໂ້', 'ໄີ້', 'ສຶ', 'ງັຸ', 'ໄ່', 'ຊັູ້', 'ຣ້ໍ', 'ຈິ້', 'ລົ', 'ຢັ໊',
     'ະົ', 'ຜືູ', 'ຊຼ', 'ປ໋', 'ຢັຼ້', 'ຊັູ່', 'ຍຶ', 'ປີຸ່', 'ທ້໌', 'ຮີ່', 'ຊີູ້', 'ດ໋', 'ມີ', 'ຕືຼ້', 'ມັຸ', 'ຮັ້',
     'ຮ້ໍ', 'ຕຼ', 'ຮ໋', 'ມີ້', 'ປຶ້', 'ຍິ່', 'ຍູ່', 'ຜິ້', 'ວ້ໍ', 'ວັູ້', 'ຫຼ', 'ຢ໊', 'ຟຼໍ', 'ເື່', 'າູ່', 'ຫ່້ໍ',
     'ຈີູ', 'ຣູ', 'ຮ່', 'ຍ້ໍ', 'ຝັູ', 'ພື', 'ໂັ້', 'ຕຼ້', 'ພຶ', 'ປັ້', 'ຜູ້ໍ', 'ຊິູ', 'ຜູ້', 'ລືູ', 'ຊິຼ', 'ຕັ໋', 'ກີຼ',
     'ກີູ້', 'ຂໍ', 'ຢູ່ໍ', 'ຄີ່', 'ອຶູ', 'ຕຶ໊', 'ຖື', 'ຢຸ', 'ຮີຸ້', 'ຕ໌', 'ຫ່', 'ຣີ່', 'ຄຶ່', 'ວົ່', 'ຊົ໋', 'ລ່້',
     'ດຶ໋', 'ທໍ', 'ພືູ', 'ບຸ໋', 'ຍຶ້', 'ຖໍ', 'ຄັຼ້', 'ຫຸ້ໍ', 'ຕຶ໋', 'ມ໊ໍ', 'ອ້', 'ຽ໋', 'ຕີ່້', 'ນຶ໋', 'ຈືຼ', 'ບູ',
     'ຍືູ', 'ຫັຼ້', 'ຮ໌', 'ຢູົ', 'ຈືຸ', 'ງ້ໍ', 'າົ', 'ຊັ່', 'ບີ໋', 'ຫີ້', 'ນື໊', 'ລ້', 'ຝູ', 'ຊ່', 'ຈີ້', 'ປົ', 'ຣັ',
     'ມິ່', 'ປິ໋', 'ກື້', 'ຊຶ່', 'ຊືຼ້', 'ກັຸ', 'ຄຼໍ', 'ນັູ', 'ວື', 'ອືຼ', 'ຊັ້', 'ສໍ', 'ຄື', 'ສູ່', 'ໃູ້', 'ດຸ', 'ປຶ໊',
     'ແ້', 'ໂຼ', 'ສົຼ', 'ດ່໋', 'ໄື່', 'ສ໊', 'ຣິ່', 'ປື້', 'ຕືູ້', 'ຈິ໊', 'ອົ໋', 'ວ໌', 'ຊ້', 'ຜ່ໍ', 'ທຸ້', 'ດຶ່', 'ນູ້ໍ',
     'ລ໊', 'ຍັຼ້', 'ຈ໋ໍ', 'ະີ', 'ຫືູ', 'ຣົ້', 'ຕົ້', 'ຄ້ໍ', 'ຜົ່້', 'ປີຼ', 'ຮື', 'ຖົ່', 'ຝີ່', 'າົຼ', 'ລົ໋', 'ປູ',
     'ປີ໋', 'ອຼ', 'ປຸ໋', 'ຈຸ້', 'ຊໍ', 'ສີູ', 'ຕ໋ໍ', 'ຢ່', 'ຄັ', 'ດື', 'ງ່', 'ຜ້ໍ', 'ຍຸ່', 'ຫ່ໍ', 'ນຶ', 'ກ່້ໍ', 'ຊືູ້',
     'ຜໍ', 'ຟີູ', 'ຊ່ໍ', 'ປຸົ', 'ກ່ໍ', 'ຖຸ໋', 'ກຶ໋', 'ບຼ', 'ຊູ່', 'ນ໊', 'ຽັ', 'ທູ', 'ນືູ້', 'ກືູ', 'ຊຸ໋', 'ຫື່', 'ຜ່້ໍ',
     'ຟຼ', 'ມິູ', 'ຖື່', 'ຊືູ່', 'ດູ', 'ຜູ່້', 'ຮູ໋', 'ຮຸໍ', 'ອຸົ', 'ຟິ', 'ຈັ', 'ຂຶຼ', 'ງື່', 'ພືູ່', 'ຄົຼ', 'ສິ່',
     'ຫັ', 'ຣີ້', 'ຟ໌', 'ດຸ໋', 'ກຶຸ', 'ທຸ', 'ທັຼ', 'ຝິຼ', 'ພຸ່', 'ຮຶ່', 'ພຸ໋', 'ຈຶ້', 'ຣ໌', 'ຕິູ', 'ປືຸ', 'ຮູ່ໍ',
     'ຈູົ້', 'ດ໌', 'ນີຼ້', 'ຟຸ໋', 'ກັູ', 'ກີ', 'ປິູ່', 'ສິູ່', 'ກື່', 'ໂິ', 'ຫຼ້', 'ອີ໊', 'ຝິ້', 'ຫີຼ', 'ໃັ', 'ປິ໊',
     'ທິຼ', 'ຮື໋', 'ຈື້', 'ຮັູ', 'ຮີູ', 'ລືຼ່', 'ຣ໋', 'ວັູ', 'ກ໊ໍ', 'ທີູ່', 'ລັ່', 'ຄ່ໍ', 'ຕົ໋', 'ຢືູ', 'ຈຸ໋', 'ຢຸ່',
     'ພູໍ', 'ຊັູ', 'ປູ່ໍ', 'ທຶ', 'ຂຶ', 'ຖຶ', 'ຣື້', 'ຫິ້', 'ຕຸ່', 'ພັ້', 'ຂັ້', 'ມຼ', 'ວັຼ', 'ຍ່', 'ປືຸ້', 'ເີ່', 'ມ່ໍ',
     'ຫ້ໍ', 'ປືຸ່', 'ສູ້', 'ພຸ້', 'ພິ', 'ບັ', 'ຂົ່', 'າ່ໍ', 'ຕັ່້', 'ຝູ໋', 'ຂຸໍ', 'ດ້ໍ', 'ສິູ', 'ຍື', 'ນັ່', 'ກຸົ',
     'ຍໍ', 'ງຸ້', 'ຊົຼ', 'ຄ໌', 'ຮຸ໋', 'ມີຼ', 'ຢູ່', 'ຢຶ', 'ນົ໋', 'ມີຸ່', 'ຢ໋', 'ຫັ້', 'ອືຼ່', 'ຄິ່', 'ມີ໋', 'ກືຸ່',
     'ຢ້', 'ຢັ້', 'ວັ້', 'ນຼ', 'ວ້', 'ບົຼ', 'ຢູ່້', 'ນັຼ', 'ປົ່້', 'ງູ້', 'ຕົ', 'ຄືຼ່', 'ກູ້', 'ຕືຼ', 'ຖ້', 'ຮິ', 'ຟ໊',
     'ສິຼ', 'ກຼໍ', 'ປູ່', 'ວິ່', 'ດັ', 'ຢັຼ', 'ໃ່ໍ', 'ຫຸ', 'ຜູໍ', 'ຜືູ່', 'ບູ່ໍ', 'ນີຸ', 'ທ໌', 'ຕີ້', 'ອຶ', 'ພິ່້',
     'ງືຼ', 'ໄ້ໍ', 'ຖຶ້', 'ຈ້', 'ຮັ່', 'ຄຶ້', 'ລັຸ', 'ໂີ', 'ຈູ່', 'ຜຸົ', 'ປົ໊', 'ເູ້', 'ຊື່', 'ຢຶ້', 'ບິຼ', 'ມູ', 'ໄີ',
     'ຣູ້', 'ຢິ໊', 'ຄີຼ', 'ຝົ້', 'ລຶ', 'ຣື່', 'ວ໊', 'ຍັຼ', 'ງ໌', 'ຟືຸ', 'ຄິູ', 'ຂ໊', 'ລົຼ', 'າຶ', 'ໄ່ໍ', 'ຽ້', 'ງູ່',
     'ຂູໍ', 'ກຸ໋', 'ຊິູ່', 'ຕືຸ້', 'ອິ໋', 'ຝີ', 'ຕູົ້', 'ຈັ໋', 'ນັຸ້', 'ນູ໊', 'ຝູ່', 'ບິູ່', 'າີ່', 'ຟ໋', 'ບ໋ໍ', 'ນຸົ',
     'ເູ່', 'ຢຼ', 'ຮັຸ', 'ບ໋', 'ສື່້', 'ຕຶ້', 'ບ້ໍ', 'ຕືູ', 'ອູ', 'ທິ້', 'ຮີຸ່', 'ຂັຼ້', 'ຄືູ່', 'ກັ໊', 'ຂ໌', 'ວູົ້',
     'ຂຸ່', 'ສືູ', 'ງູ໋', 'າິຸ', 'ຊັຼ', 'ຊຸ່', 'ລູໍ', 'ດ່ໍ', 'ອ໋', 'ສົ', 'ນື້', 'ຕຼ໌', 'ຂືຼ', 'ລີຸ່', 'ຢິູ່', 'ວຶ່',
     'ສືູ້', 'ວັ່', 'ມົ', 'ຈຸ້ໍ', 'ປີຸ', 'ຈ່້', 'ຈີ', 'ລຸ່', 'ໄໍ', 'ດົ່', 'ມັຸ້', 'ຄັ່້', 'ຄັ່', 'ມີ່', 'ຂິ່', 'ມູ໊',
     'ຍຸ', 'ມ໋ໍ', 'ຖ໌', 'ຝິ່', 'ພີ້', 'ວັ໌', 'ມັູ້', 'ມື໊', 'ຄຸ໋', 'ມິ', 'ສຸົ້', 'ດີຸ', 'ຫືຼ່', 'າົ້', 'ກ່', 'ດີຼ',
     'ຜີ', 'ຣໍ', 'ຊຼ໌', 'ໃ້', 'ຄີູ້', 'ຍືູ້', 'ຍຸ້', 'ຮຼ່', 'ະ້', 'ງົ', 'ພ່້', 'ຮື້', 'ຂຶ້', 'ຟ໋ໍ', 'ຍີູ້', 'ຊຸ້', 'ມື',
     'ກຸ້', 'ນືຼ້', 'ນັ່້', 'ຊູ', 'ຫຼ໋', 'ດັ໋', 'ຜຸົ້', 'ຣື່້', 'ຖັ່', 'ຮີ້', 'ຈູ້', 'ຕັ໊', 'ອົ໊', 'ທີ່', 'ມິ໋', 'ຫີຼ້',
     'ກັຼ', 'ຈົ້', 'ປື່', 'ຄູົ່', 'ຕີ', 'ຂຼໍ', 'າ່', 'ງົຼ', 'ປັ່', 'ໃີ', 'ນ້', 'ວືຼ', 'ຕໍ', 'ຢັ', 'ຄູ', 'ນູ໋', 'ຖູ້',
     'ປິ', 'ດິຸ', 'ດິ', 'ຈິ໋', 'ຂັູ້', 'ຫົ', 'ພ໊', 'ດິ້', 'ປຸ້', 'ຣຸ', 'ບີ', 'ປີ້', 'ໄີ່', 'ຜູ່', 'ນັຼ້', 'ຢີ້', 'ຫີຼ໋',
     'ປຸ', 'ຖົ໋', 'ຊິ', 'ຍ່ໍ', 'ທິ່', 'ກົ່', 'ງູ', 'ໂ໌', 'ຊຼ່', 'ຍື໊', 'ສ້໋', 'ເ໌', 'ດ່້', 'ລຸົ', 'ຍົ', 'າໍ', 'ຈັ້',
     'ຫື່້', 'ຮູໍ', 'ນົ', 'ຍີ່', 'ຊຶ', 'ຊືູ', 'ທຸ໌', 'ປຸ໊', 'ຊື', 'ຕູົ', 'ນືູ', 'ອ໊', 'ກ໋', 'ງຸ່', 'ວູ່', 'ລຸໍ', 'າູ້',
     'ອີ໋', 'ຊຸົ້', 'ບຼ໊', 'ສູ', 'ທື', 'ຄູໍ', 'ຫຼໍ', 'ນືຼ', 'ເ່', 'ປີ່', 'ອັຼ', 'ພືຸ້', 'ຕຸ', 'ຖີ່', 'ອຸ', 'ຢຶ່',
     'ບັຸ້', 'ຖູ່', 'ຫິຼ່', 'ຢ໌', 'າື', 'ກູ້ໍ', 'ຝົ່', 'ອື່', 'ມື໋', 'ຟັ', 'ຄີູ', 'ສຼ', 'ກຸົ່', 'ຊີຸ', 'ຈືູ່', 'ກິ໊',
     'ຟິ້', 'ລຼໍ', 'ຫິ່', 'ໄ໌', 'ໂຸ', 'ນີ໊', 'ທຶ້', 'ຕິ໋', 'ຝັູ່', 'ອ໋ໍ', 'ຮຸົ', 'ຟີ', 'ວຶ້', 'ມຸ', 'ງື້', 'ຍ່້', 'ຕຸ້',
     'ຣຼ', 'ຣູ່', 'ນັູ້', 'ດູ່', 'ມຸ໋', 'ຍືຸ້', 'ນ່ໍ', 'ຕືູ່', 'ລື໊', 'ດື່', 'ຄຼ', 'ງິ່', 'ຟີຼ', 'ມັຼ', 'ບຼ໋', 'ສີູ້',
     'ເັ້', 'ຍູ', 'ທືຼ', 'ດີ້', 'ຫຶ', 'ມູ່', 'ຄິ', 'ຮູ໊', 'ອິ', 'ຟຸ', 'ຍ໌', 'ມຸ້ໍ', 'ຄົ້', 'ນືູ່', 'ລ໋', 'ດຶ', 'ຕູໍ',
     'ບຶ້', 'ບັ໊', 'ອັ່', 'ຄີູ່', 'ອຸໍ', 'ຂ໋', 'ຜື', 'ນ໋ໍ', 'າ້ໍ', 'ຫົຼ', 'ຍູ້', 'ຫິຼ', 'ຖັ໋', 'ຝື້', 'ດູ໊', 'ຂືູ',
     'ຕູ້ໍ', 'ລຸ້', 'ສູົ້', 'ສີຼ', 'ພີ່', 'ນຶູ', 'ມັູ', 'ຂຼ', 'ຕ໊', 'ໆ້', 'ຮ໊', 'ຂົ່້', 'ບຶ', 'ຜູົ', 'ຂິ', 'ນຸ່', 'ຟື',
     'ຢັູ', 'າຶ່', 'ກຸ', 'ຖິ່', 'ບື້', 'ມັ່້', 'ນິ໋', 'ຮູ', 'ບັູ', 'ງຼ', 'ລຸ້ໍ', 'ກົ່້', 'ຜ່້', 'ພູ່ໍ', 'ພື໊', 'ຫ໌',
     'ຫິ', 'ຂື້', 'ຂຶູ້', 'ຜົ', 'ອີ', 'ຄີ', 'ຮື່', 'ຫົຼ້', 'ກຶ່', 'າັ', 'ຈຶ່', 'ກິ໋', 'ວຼ່', 'ສືຼ', 'ຫົ່', 'າິ້', 'ຊິ່',
     'ບົ', 'ພູ້', 'ຖູ', 'ສື່', 'ມົ໋', 'ຊັ໋', 'ນັ໋', 'ຕຸົ', 'ຂືຸ້', 'ອັ້', 'ຄືຼ', 'ຄຼ່', 'ບ໌', 'ມືູ່', 'ທຼ', 'ຢົ້',
     'ກຼ໋', 'າິ່', 'ມື່', 'ຈື໊', 'ມັ໊', 'ະີ້', 'ພູ', 'ຖີ', 'ຂ່້ໍ', 'ອຸົ້', 'ພົ', 'ສ່', 'ວູ້', 'ນື໋', 'ສົ້', 'ພື້',
     'ຟຸ່', 'ຟ່ໍ', 'ຈຸ', 'ນີ້', 'ຜ໋', 'ເີ', 'ອຸ່', 'ຟັຼ້', 'ນຸ', 'ອຸ່ໍ', 'ະ໋', 'ນົ່', 'ນ໊ໍ', 'ທູ່', 'ຟໍ', 'ປູ໋', 'ນີຸ້',
     'ລັ', 'ກັ', 'ມັ໋', 'ຣົ', 'ຊີຸ້', 'ຟິ່', 'ກົ໊', 'ປັ', 'ຂ້', 'ຢ່ໍ', 'ນີຸ່', 'ຟີ່', 'ນຸ້', 'ສັຸ', 'ນີູ້', 'ຽູ່', 'ຢື',
     'ບູ້', 'ຍົ້', 'ວີ', 'ຣື', 'ທີ', 'ງິ', 'ກົ້', 'າີູ', 'ລືຸ', 'ຈື', 'ຮຶ', 'ຟັ່', 'ງັ່', 'ຝັ່', 'ຄູ້', 'ຂຸົ', 'ບິ່',
     'ຂູົ້', 'ທີ່້', 'ຂັູ', 'ຕູ໋', 'ກີ໊', 'ຂ່ໍ', 'ຍີູ', 'ເິ', 'ມັ', 'ຝຸ່', 'ຍີ້', 'ສັ', 'ໃູ', 'ວື່', 'ລີ້', 'ວູ໋',
     'ຫັ໋', 'ງຼ່', 'ຢູໍ', 'ຈ່', 'ມືຼ', 'ອໍ', 'ດຶ້', 'ປູໍ', 'ຍັຸ້', 'ກີຼ້', 'ຂູົ່', 'ຝັ້', 'ກົ໋', 'ຫິຼ້', 'ຮູົ', 'ຟົ',
     'ຮຸ້ໍ', 'ກິ້', 'ະຸ', 'ຟື່', 'ສູ໊', 'ະູ', 'ຕັູ', 'ປິ້', 'ຫູ່ໍ', 'ຫືຼ້', 'ບີຼ', 'ໃື', 'ຟັ້', 'ສື້', 'ວໍ', 'ຂູ້',
     'ຄຸ້ໍ', 'ຄຸົ', 'ຫຶ້', 'ຢີ່', 'ວຼ', 'ງັ', 'ນັຸ', 'ສັ່', 'າົ່', 'ນຶ໊', 'ບຼ້', 'ພ໋', 'ຮີ', 'ທົ່້', 'ພ໌', 'ຈູົ', 'ມ໊',
     'ຢີ໋', 'ສ່້ໍ', 'ຣຶ້', 'ເຼ', 'ຈີ໋', 'ສັຸ້', 'ຫຼ້ໍ', 'ງົ້', 'ລຸົ້', 'ມື່້', 'ມຸໍ', 'ສຸົ', 'ຈົ', 'ນູ້', 'ຮືຼ', 'ບັ໋',
     'ບ່້', 'ຍັຸ', 'ພິຼ', 'ຄືຸ', 'ຍັ', 'ຈືຸ່', 'ມຶ່', 'ຊື້', 'ກິ', 'ຊັ໊', 'ກໍ', 'ກົຼ', 'ຢິ້', 'ຍີ໋', 'ຈົ໊', 'ອູ໊', 'ລິ',
     'ຣຸ່', 'ອູ່', 'ພົ້', 'ທື່', 'ຢີູ່', 'ຮໍ', 'ລີ່', 'ກູໍ', 'ກິຸ', 'ປ້', 'ໆີ່', 'ກັ້', 'ຖິູ່', 'ຈ່ໍ', 'າ້', 'ຊີ້',
     'ປ໊', 'ຜູ', 'ພ້ໍ', 'ປິຼ', 'ອີ້', 'ສຶ່', 'ດິ່້', 'ຖີ້', 'ທັ່', 'ບິ', 'ງ່ໍ', 'ລ່', 'ພ້', 'ຈື່', 'ຮົ້', 'ຢູ໋', 'ລ້ໍ',
     'ຫີຼ່', 'ຍັ້', 'ນີຼ', 'ຫ໊', 'ບ້', 'ຄຸໍ', 'ກິ່', 'ວູ', 'ຈ໋', 'ຈັ່', 'ກິຸ່', 'ງຶ້', 'ສ໋', 'ຕ່ໍ', 'ນ່້', 'ລິ່',
     'ລຸົ່', 'ສຸ້', 'ຮົ່', 'ພີູ່', 'ຄົ່', 'ຍຶ່', 'ນີ', 'ງຸ', 'ມຶ', 'ນຸ໊', 'ຕື໊', 'ຍົ່', 'ວ່ໍ', 'ຕັູ້', 'ະ່', 'ະື',
     'ຫຶຼ', 'ນຶ້', 'ຕ້ໍ', 'ຄື່', 'ຕີ໋', 'ຫໍ', 'ມີູ', 'ຕູ່ໍ', 'ທຶ່', 'ຂ້໋', 'ກຶຼ', 'ແ່', 'ຊີ໋', 'ນີ່', 'າຸ້', 'ງຶ່໋',
     'ຝິ', 'ຊິ໋', 'ທ໋ໍ', 'ຢີ', 'ວຶ', 'ພີ໊', 'ພ່ໍ', 'ຊຶ້', 'ສຶ້', 'ະິ', 'ເຸ', 'າີ້', 'ຫິຸ', 'ຣີ', 'ຫູ່', 'ນຸ໋', 'ບ່໋ໍ',
     'ຮຶ້', 'ຄິ້', 'ຽີ', 'ລືຼ', 'ກີ໋', 'ຕິ່', 'ມິ້', 'ຢືູ່', 'ບີ້', 'ຈິ່', 'ທຸ່', 'ລີ', 'ຖຸ', 'ຮູ້ໍ', 'ລື້', 'ຝຶ່',
     'ຝຼ', 'ດຶ໊', 'ຍ໊ໍ', 'ດົ໋', 'ຄື໋', 'ຖຸ້', 'ຍູົ', 'ກຸ່', 'ດ໊', 'ຈຶ', 'ຢົ່', 'າິ', 'ຊີູ', 'ຣ໊', 'ທີ້', 'ວິ້', 'ບົ້',
     'ບິ້', 'ຊ໋', 'ກູົ', 'ພຼ', 'າີູ້', 'ປັຸ', 'ຕ່້', 'ຢິ່', 'ຣິ້', 'ປັຸ່', 'ຂື່', 'ປິູ', 'ກືູ້', 'ຢູ້', 'ຍິ້', 'ຜີູ້',
     'ຣຸ໊', 'ອີ່', 'ທືູ່', 'ຟັຼ', 'ຕື', 'ຊັ່້', 'ຣືຼ', 'ຝົ', 'ແູ້', 'ທົ', 'ຄູົ້', 'ນຶູ່', 'ຊືຸ', 'ຖ້ໍ', 'ນິ່', 'ຊຸ',
     'ພ່', 'ຝຶ', 'ພົ່', 'ອຸ໊', 'ຢີູ', 'ຄີຸ', 'ກູ໋', 'ມັ້', 'ງື໋', 'ທີຼ', 'ຮູ່້', 'ສີຸ', 'ຢັູ້', 'ກູ', 'ຕັຸ້', 'ປົ້',
     'ຂ້ໍ', 'ຍີ', 'ຄິຼ່', 'ລຸ໋', 'ຫົ້', 'ກ້໋', 'ສ່້', 'ໆື່', 'ບັ່', 'ຂຶຸ້', 'ສຸ່', 'າີ', 'ບິ໊', 'ຕຶ', 'ດຸ້', 'ຖົ',
     'ດຼ້', 'ຕີຼ', 'ະັ້', 'ອົ', 'ຄື້', 'ຄິຼ', 'ຝື່', 'ຫື້', 'ມີູ່', 'ຂີ່', 'ຖູົ່', 'ຫຸໍ', 'ມຶ້', 'ຢື໋', 'ຮັຸ້', 'ຈໍ',
     'ຊູ້', 'ປື', 'ຢູົ່', 'ຂູ', 'ອູ໋', 'ຝຸ້', 'ຂຸ້ໍ', 'ທຸົ', 'ຂືູ່', 'ລັຼ້', 'ຍື່', 'ຜຶ້', 'ຝຸ', 'ຽູ', 'ກືຼ', 'ຕຼ່',
     'ທ໊', 'ປ່ໍ', 'ຢຸ໋', 'ຝັຼ', 'ງຶ', 'ທົ໋', 'ຜ໌', 'ວຸົ', 'ລັຼ', 'ຂູ້ໍ', 'ຫູ໋', 'ຊັ', 'ຮ່້', 'ມິ໊', 'ສັ້', 'ດັຼ້', 'ໂໍ',
     'ຣົ່', 'ທູົ', 'ຣ້', 'ພຸ', 'ຂັ', 'ມິຸ', 'າ໊', 'ຄັຸ້', 'ຄຶ', 'ບ່', 'ຮັ', 'ທີູ', 'ຟູ໋', 'ທີຼ່', 'ວັ໋', 'ມີ໊', 'ຍິ໊',
     'ປູ໊', 'ຕິ', 'ງິ໋', 'ຄົ', 'ບູໍ', 'ອືູ', 'ຊັຸ້', 'ຢິ໋', 'ງີ່', 'ຢິູ', 'ໄິ', 'ປຸົ້', 'ທິ', 'ທັ່້', 'ກຸ໊', 'ບໍ',
     'ເີ້', 'ຮືູ', 'ທ່', 'ຍືຸ', 'າ່້', 'ປຶ່', 'ຕຸ໊', 'ປິຼ່', 'ຟູ່', 'ບັູ້', 'ຣຶ່', 'ຮີຸ', 'ຟູ', 'ມ່', 'ຫ໋', 'ຖົ້',
     'ຕິ໊', 'ຜຶ', 'ຄູ່ໍ', 'ມົ້', 'ຖີ໋', 'ລືຸ່', 'ຄັຸ', 'ຂຸົ້', 'ປ໊ໍ', 'ດີ່້', 'ໃຼ', 'ລົຼ່', 'ລຶ້', 'ຊ໌', 'ມຶ໋', 'ຫຶຼ້',
     'ຜິ', 'ຂັຸ້', 'ດົ໊', 'ຟີູ້', 'ພູ່', 'ໄຼ', 'ກຶ', 'ຜື່', 'ເັ', 'ພື່້', 'ຄັູ້', 'ຈູ', 'ຈຸ່', 'ອື້', 'າ໋', 'ຜຼ້',
     'ຖິ້', 'ຍື້', 'ນຸ້ໍ', 'ຫູ້ໍ', 'ເ໋', 'ຜີ່', 'ຖຸ່', 'າັ້', 'ຄຸ້', 'ໄູ', 'ງີ໊', 'ຮຸ່', 'ຢ້ໍ', 'ກ໊', 'ນຸໍ', 'ນຶ່',
     'ຣິ', 'ຮີູ້', 'ຫຸ່', 'ພິ້', 'ກັູ້', 'ວີ້', 'ສັູ່', 'ດິ່', 'ຄູ໋', 'ຄືຸ້', 'ບົ໋', 'ຣັ່', 'ຈີູ່', 'ຫີ', 'ທູ້', 'ສີ່',
     'ແັ', 'ຈົ່້', 'ອຶ້', 'ລຼ', 'ຂືຼ້', 'ຮຼ້', 'ສື', 'ຢື່', 'ທັ້', 'ນົ້', 'ຖຶ່', 'ນູໍ', 'ບູ່', 'ຮັຼ', 'ຂືຸ', 'ຝຼ່',
     'ຜຶ່', 'ປົ່', 'ບຸ', 'ຄ້', 'ມຸ໊', 'ຕູ໊', 'ຖິູ', 'ປຸ່', 'ອິ້', 'ບູ໊', 'ງັຼ', 'ຮົ', 'ຂຸ', 'ຟ້ໍ', 'າົຼ້', 'ມືຸ້',
     'ຮິ້', 'ຕັຸ', 'ຜັ້', 'ລິ້', 'ມືຸ', 'ຜຸ', 'ອົ່', 'ະັ', 'ວຸົ້', 'ຈັຼ້', 'ຟຸ້', 'ນໍ', 'ອູ້', 'ສ້', 'ມົ່', 'ຂີ', 'ລັ໊',
     'ດ່', 'ຜີູ', 'ທື້', 'ຫູ່້', 'ນິ້', 'ລົ້', 'ບິ໋', 'ຊືຸ້', 'ຮູ້', 'ຢົ໊', 'ຫຸ້', 'ນ້ໍ', 'ປ໌', 'ຈຸໍ', 'ມຼ່', 'ກູ່',
     'ສິ', 'ເູ', 'ຕີູ', 'ປ້ໍ', 'ດັຼ', 'ສີຸ່', 'ຕື້', 'ບົ໊', 'ຕຸົ້', 'ຕ໋', 'ດືຼ', 'ບັຸ', 'ຂຶຸ', 'ຜຸ້', 'ຟົ້', 'ຊູົ',
     'ຟູ້', 'ຕື໋', 'ຍັ່', 'ມ໌', 'ຖື໋', 'ຖ່ໍ', 'ຽ່', 'ຟົຼ', 'ຝີ້', 'ສູົ່', 'ໆີ', 'ລຸ', 'ກືຸ', 'ຜຼ', 'ຄື໊', 'ຜຸ່', 'ດົ',
     'ທ່ໍ', 'ຜິຼ', 'ລ່ໍ', 'ງ໊', 'ບຸ່', 'ຄ໊', 'ພີຼ', 'ຕົ໊', 'ຈ໌', 'ປໍ', 'ນື', 'ກັ໋', 'ໃ່', 'ຄິ໋', 'ຣຶ', 'ໆ່', 'ຕ້', 'ຈິ',
     'ພຶ້', 'ງື', 'ຜັ', 'ບິູ', 'ຊິ້', 'ອຶ່', 'ລີ໋', 'ໆື', 'ບຼ່', 'ບຸ໊', 'ໄູ້', 'ຢໍ', 'ທືູ', 'ສ່ໍ', 'ຍຸ້ໍ', 'ນືຸ', 'ອ໌',
     'ຂົ', 'ນິ', 'ດໍ', 'ມ້ໍ', 'ພັຼ', 'ຄຼ້ໍ', 'ຫ້', 'ອື', 'ຂີ້', 'ຟ່', 'ງຶ່', 'ມັຼ້', 'ງືູ', 'າົຼ່', 'ທັ໋', 'ປີ', 'ຫັຼ່',
     'ຜ່', 'ວິ', 'ບົ່', 'ຝູ້', 'ດີ່', 'ວົ', 'ວູົ', 'ຖ່', 'ຮຸ້', 'ດີຸ່', 'ປັ໊', 'ຕັຼ', 'ຣ່', 'ສູົ', 'ໄ້', 'ພິ່', 'ດ້',
     'ຕືຸ', 'ຟຶ', 'ຖັຼ້', 'ຊົ່', 'ຫຸົ', 'ຜັຼ', 'ນັ', 'ວີ່', 'ຽ໌', 'ນ໋', 'ສຸ', 'ຝັ', 'ນັ້', 'ລື', 'ອັ໊', 'ຂົ້', 'ທຼ່',
     'ວ່', 'ນ່', 'ມໍ', 'ຄຸົ້', 'ຖັຼ', 'ຢື໊', 'ເື', 'ຫົຼ່', 'ລຶ່', 'ຕູ', 'ມື້', 'ຫູໍ', 'ຕຶ່', 'ບື່', 'ວັຸ', 'ລີຸ', 'ຮຸ',
     'ຫູ້', 'ນືຼ່', 'ຫັ່', 'າື່', 'ຢືຼ', 'ພັ', 'ເຶ', 'ແີ', 'ງັ້', 'ຍິ', 'ຫຶ່', 'ກັ່', 'ກຼ່', 'ດື້', 'ມ໋', 'ສີ້', 'ຟີ້',
     'ທູົ່', 'ອັ', 'ນົຼ', 'ຍຼ', 'ຈີ່', 'ຮືູ້', 'ຜູົ້', 'ພັ່', 'ນີ໋', 'ດູ້', 'ປີ໊'])


def parse_text(text):
    arr = []
    i = len(text) - 1
    while i >= 0:
        if text[i] not in main_character and text[i] not in other_character:
            if i > 0 and std_compound(text[i - 1:i + 1])[1] in compounds:
                arr.append(std_compound(text[i - 1:i + 1])[1])
                i -= 2
            elif i > 1 and std_compound(text[i - 2:i + 1])[1] in compounds:
                arr.append(std_compound(text[i - 2:i + 1])[1])
                i -= 3
            elif i > 2 and std_compound(text[i - 3:i + 1])[1] in compounds:
                arr.append(std_compound(text[i - 3:i + 1])[1])
                i -= 4
            else:
                # print("-------error-----")
                # print(text)
                # print(str(i))
                # print(text[i])
                i -= 1
        else:
            arr.append(text[i])
            i -= 1
    # print(arr)
    arr.reverse()
    return arr


def is_valid(text):
    is_valid = True

    i = len(text) - 1
    while i >= 0:
        if text[i] not in main_character and text[i] not in other_character:
            if i > 0 and std_compound(text[i - 1:i + 1])[0] and std_compound(text[i - 1:i + 1])[1] in compounds:
                i -= 2
            elif i > 1 and std_compound(text[i - 2:i + 1])[0] and std_compound(text[i - 2:i + 1])[1] in compounds:
                i -= 3
            elif i > 2 and std_compound(text[i - 3:i + 1])[0] and std_compound(text[i - 3:i + 1])[1] in compounds:
                i -= 4
            else:
                is_valid = False
                break
        else:
            i -= 1
    return is_valid


def std_compound(compound):
    is_valid = True
    newcompound_arr = [compound[0]]

    for i in range(1, len(compound)):
        # if compound[i] not in newcompound_arr and compound[i] != "່" and compound[i] != "້" and compound[i] != "໋" and \
        #         compound[i] != "໊":
        if compound[i] not in newcompound_arr:
            newcompound_arr.append(compound[i])
    newcompound = "".join(newcompound_arr)

    newcompound = newcompound[0] + "".join(sorted(split(newcompound[1:])))

    for rule in invalid_rule:
        if rule[0] in newcompound and rule[1] in newcompound:
            is_valid = False

    # for i in range(0, 5):
    for rule in rule_top:
        # print(rule)
        while rule[0] in newcompound and rule[1] in newcompound:
            newcompound = newcompound.replace(rule[0], "")

    return is_valid, newcompound


# Python3 program to Split string into characters
def split(word):
    return [char for char in word]


if __name__ == '__main__':
    print(is_valid("ໍາຫຶ"))
    print(parse_text("ໍາຫຶ"))
    print(parse_text("ອິນປັນຍາ"))
    # #
    # first_list = set(consonants)
    #
    # sign_list = set()
    #
    # count = 0
    #
    print(len(compounds))

    sorted_compounds_set = set()
    for compound in compounds:
        sorted_compounds_set.add(compound[0] + "".join(sorted(split(compound[1:]))))
    print(len(sorted_compounds_set))
    print(sorted_compounds_set)
    #     first_list.add(compound[0])
    #     for i in range(1, len(compound)):
    #         if compound[i] not in first_list:
    #             sign_list.add(compound[i])
    #
    #     for i in range(1, len(compound)):
    #         for j in range(i + 1, len(compound)):
    #             if compound[i] == compound[j]:
    #                 print(compound)
    #                 count += 1
    #                 break
    #
    # print(len(first_list))
    # print(len(sign_list))
    # print(count)
    #
    # print(first_list)
    # print(sign_list)
    #
    # print(std_compound("ລືໍ"))
    # arr = []
    #
    # for compound in compounds:
    #     if len(compound) > 1:
    #         arr.append(compound)
    #
    # print(arr)

    # ດິ
    # ດຶ
    # ດົ
    # ດ້
    # ກີ
    # ດື
    # ດູ
    # ດໍ