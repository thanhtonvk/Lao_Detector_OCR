from enum import Enum


class CardType(Enum):
    FRONT_ID = "front_id"
    FRONT_WHITE_ID = "front_white_id"
    FRONT_GREEN_ID = "front_green_id"
    BACK_ID = "back_id"
    PASSPORT = "passport"
    HOUSEHOLD_HANDWRITING = "handwriting_household"
    HOUSEHOLD_PRINT = "print_household"
    HOUSEHOLD_NEXT_HANDWRITING = "handwriting_household_next"
    HOUSEHOLD_NEXT_HANDWRITING_ROW = "handwriting_row"
    HOUSEHOLD_NEXT_PRINT = "print_household_next"
    MISSING_INFORMATION = "missing_information"
    BLACK_AND_WHITE = "black_and_white"
    # BLUR_IMAGE = "blur_image"
    # TEXT_BLUR = "text_blur"
    GLARE = "glare"
    PPI = 'ppi'
    BRIGHTNESS = "brightness"
    DARKNESS = 'darkness'
    MULTI_CARD = "multi_card"
    BLUR_CARD = "is_blur_card"

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

    @classmethod
    def get_enum(cls, value):
        obj = cls._value2member_map_
        return obj[value] if value in obj else None
