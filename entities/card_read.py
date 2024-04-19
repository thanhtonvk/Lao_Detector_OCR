from pydantic import BaseModel
from entities.card_type import CardType
from typing import List


class CardField(BaseModel):
    value: str
    conf: float


class CardReadFields(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        if key not in self.__dict__:
            raise AttributeError(
                "'{}' object has no attribute '{}'".format(self.__class__.__name__, key)
            )
        self.__dict__[key] = value


class IdCardReadFields(CardReadFields):
    id_number: CardField = None
    name: CardField = None
    birthday: CardField = None
    place_of_birth: CardField = None
    address: CardField = None
    district: CardField = None
    province: CardField = None
    ethnic: CardField = None
    nationality: CardField = None
    occupation: CardField = None
    issue_place: CardField = None
    issue_date: CardField = None
    expiry: CardField = None


class PassportReadFields(CardReadFields):
    type: CardField = None
    state: CardField = None
    id: CardField = None
    sur_name: CardField = None
    given_name: CardField = None
    name_laos: CardField = None
    nationality: CardField = None
    gender: CardField = None
    dob: CardField = None
    place_of_birth: CardField = None
    issue_date: CardField = None
    expire_date: CardField = None
    authority: CardField = None
    personal_code: CardField = None
    name: CardField = None
    sex: CardField = None
    country_code: CardField = None
    nationality_code: CardField = None
    code: CardField = None


class HouseholdReadFields(CardReadFields):
    name: CardField = None
    ward: CardField = None
    district: CardField = None
    province: CardField = None
    total: CardField = None
    male: CardField = None
    female: CardField = None
    issue_date: CardField = None
    street: CardField = None
    unit: CardField = None
    block: CardField = None
    number: CardField = None
    id: CardField = None
    lane: CardField = None
    house_holder_name: CardField = None


class HouseholdNextReadFields(CardReadFields):
    name: CardField = None
    dob: CardField = None
    relationship: CardField = None
    ethnic: CardField = None
    occupation: CardField = None
    race: CardField = None
    nationality: CardField = None
    workplace: CardField = None
    hometown: CardField = None
    relocate: CardField = None
    male: CardField = None
    female: CardField = None
    id_number: CardField = None
    id: CardField = None
    religion: CardField = None
    gender: CardField = None


class CardRead(BaseModel):
    label: CardType = None
    score: float = None
    content: CardReadFields


class IdCardRead(CardRead):
    content: IdCardReadFields
    is_grey: bool = False


class PassportRead(CardRead):
    content: PassportReadFields


class HouseholdRead(CardRead):
    content: HouseholdReadFields


class HouseholdNextRead(CardRead):
    content: HouseholdNextReadFields
