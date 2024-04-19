import numpy as np
from typing import List
from pydantic import BaseModel
from enum import Enum


class CardDetectionFields(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    def __setitem__(self, key, value):
        if key not in self.__dict__:
            raise AttributeError(
                "'{}' object has no attribute '{}'".format(self.__class__.__name__, key)
            )
        self.__dict__[key] = value


class IdCardDetectionFields(CardDetectionFields):
    id_number: List[np.ndarray] = None
    name: List[np.ndarray] = None
    birthday: List[np.ndarray] = None
    place_of_birth: List[np.ndarray] = None
    address: List[np.ndarray] = None
    district: List[np.ndarray] = None
    province: List[np.ndarray] = None
    ethnic: List[np.ndarray] = None
    nationality: List[np.ndarray] = None
    occupation: List[np.ndarray] = None
    issue_place: List[np.ndarray] = None
    issue_date: List[np.ndarray] = None
    expiry: List[np.ndarray] = None


class PassportDetectionFields(CardDetectionFields):
    type: List[np.ndarray] = None
    state: List[np.ndarray] = None
    id: List[np.ndarray] = None
    sur_name: List[np.ndarray] = None
    given_name: List[np.ndarray] = None
    name_laos: List[np.ndarray] = None
    nationality: List[np.ndarray] = None
    gender: List[np.ndarray] = None
    dob: List[np.ndarray] = None
    place_of_birth: List[np.ndarray] = None
    issue_date: List[np.ndarray] = None
    expire_date: List[np.ndarray] = None
    authority: List[np.ndarray] = None
    code_1: List[np.ndarray] = None
    code_2: List[np.ndarray] = None
    code_3: List[np.ndarray] = None
    code_4: List[np.ndarray] = None
    code_5: List[np.ndarray] = None


class HouseholdDetectionFields(CardDetectionFields):
    id: List[np.ndarray] = None
    unit: List[np.ndarray] = None
    ward: List[np.ndarray] = None
    district: List[np.ndarray] = None
    province: List[np.ndarray] = None
    total: List[np.ndarray] = None
    female: List[np.ndarray] = None
    issue_date: List[np.ndarray] = None
    number: List[np.ndarray] = None
    lane: List[np.ndarray] = None
    street: List[np.ndarray] = None
    block: List[np.ndarray] = None
    male: List[np.ndarray] = None
    name: List[np.ndarray] = None


class HouseholdHandwritingDetectionFields(CardDetectionFields):
    number: List[np.ndarray] = None
    street: List[np.ndarray] = None
    unit: List[np.ndarray] = None
    block: List[np.ndarray] = None
    ward: List[np.ndarray] = None
    district: List[np.ndarray] = None
    province: List[np.ndarray] = None
    total: List[np.ndarray] = None
    male: List[np.ndarray] = None
    female: List[np.ndarray] = None
    issue_date: List[np.ndarray] = None
    name: List[np.ndarray] = None


class HouseholdNextHandwritingDetectionFields(CardDetectionFields):
    name: List[np.ndarray] = None
    male: List[np.ndarray] = None
    dob: List[np.ndarray] = None
    relationship: List[np.ndarray] = None
    race: List[np.ndarray] = None
    nationality: List[np.ndarray] = None
    ethnic: List[np.ndarray] = None
    occupation: List[np.ndarray] = None
    workplace: List[np.ndarray] = None
    hometown: List[np.ndarray] = None
    relocate: List[np.ndarray] = None
    female: List[np.ndarray] = None
    id_number: List[np.ndarray] = None


class HouseholdNextPrintDetectionFields(CardDetectionFields):
    name: List[np.ndarray] = None
    dob: List[np.ndarray] = None
    relationship: List[np.ndarray] = None
    race: List[np.ndarray] = None
    nationality: List[np.ndarray] = None
    ethnic: List[np.ndarray] = None
    religion: List[np.ndarray] = None
    occupation: List[np.ndarray] = None
    id: List[np.ndarray] = None


class CardRequiredFields(Enum):
    ID_CARD = [
        "id_number",
        "name",
        "birthday",
        "address",
        "district",
        "province",
        "ethnic",
        "nationality",
        "issue_date",
        "expiry",
    ]
    PASSPORT = [
        "type",
        "state",
        "id",
        "sur_name",
        "given_name",
        "name_laos",
        "nationality",
        "gender",
        "dob",
        "place_of_birth",
        "issue_date",
        "expire_date",
        "authority",
    ]
    HOUSEHOLD_HANDWRITING = [
        # "ward",
        # "district",
        # "province",
        # "total",
        # "male",
        # "female",
        "issue_date",
        "name",
    ]
    HOUSEHOLD_PRINT = [
        "id",
        "unit",
        "ward",
        "district",
        "province",
        "total",
        "female",
        "issue_date",
    ]
    HOUSEHOLD_NEXT_HANDWRITING = ["name", "dob"]
    HOUSEHOLD_NEXT_PRINT = ["name", "dob"]


class CardDetection(BaseModel):
    content: CardDetectionFields = None
    required_keys: CardRequiredFields = None

    def missing_keys(self) -> List[str]:
        missing_keys = []
        content_keys = self.content.dict(exclude_none=True).keys()
        required_keys = self.required_keys.value
        if required_keys and len(required_keys) > 0:
            # content_keys = self.content_keys()
            for key in required_keys:
                if key not in content_keys:
                    missing_keys.append(key)
        return missing_keys


class IdCardDetection(CardDetection):
    content: IdCardDetectionFields
    required_keys = CardRequiredFields.ID_CARD


class PassportDetection(CardDetection):
    content: PassportDetectionFields
    required_keys = CardRequiredFields.PASSPORT


class HouseholdHandwritingDetection(CardDetection):
    content: HouseholdHandwritingDetectionFields
    required_keys = CardRequiredFields.HOUSEHOLD_HANDWRITING


class HouseholdPrintDetection(CardDetection):
    content: HouseholdDetectionFields
    required_keys = CardRequiredFields.HOUSEHOLD_PRINT


class HouseholdNextHandwritingDetection(CardDetection):
    content: HouseholdNextHandwritingDetectionFields
    required_keys = CardRequiredFields.HOUSEHOLD_NEXT_HANDWRITING


class HouseholdNextPrintDetection(CardDetection):
    content: HouseholdNextPrintDetectionFields
    required_keys = CardRequiredFields.HOUSEHOLD_NEXT_PRINT
