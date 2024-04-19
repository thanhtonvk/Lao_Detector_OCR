import numpy
import cv2
from abc import ABC, abstractmethod
from entities.card_detection import (
    CardDetection,
    IdCardDetection,
    PassportDetection,
    HouseholdHandwritingDetection,
    HouseholdPrintDetection,
    HouseholdNextHandwritingDetection,
    HouseholdNextPrintDetection,
)


class Detector(ABC):
    @abstractmethod
    # bgr format
    def detect_fields(self, cropped_image: numpy.ndarray) -> CardDetection:
        pass

    @staticmethod
    def get_concat_h_multi_blank(im_list, interpolation=cv2.INTER_CUBIC):
        h_max = max(im.shape[0] for im in im_list)
        im_list_resize = [
            cv2.resize(
                im,
                (int(im.shape[1] * h_max / im.shape[0]), h_max),
                interpolation=interpolation,
            )
            for im in im_list
        ]
        return cv2.hconcat(im_list_resize)


class IdCardDetector(Detector):
    @abstractmethod
    # bgr format
    def detect_fields(self, cropped_image: numpy.ndarray) -> IdCardDetection:
        pass


class PassportDetector(Detector):
    @abstractmethod
    # bgr format
    def detect_fields(self, cropped_image: numpy.ndarray) -> PassportDetection:
        pass


class HouseholdHandwritingDetector(Detector):
    @abstractmethod
    # bgr format
    def detect_fields(
        self, cropped_image: numpy.ndarray
    ) -> HouseholdHandwritingDetection:
        pass


class HouseholdPrintDetector(Detector):
    @abstractmethod
    # bgr format
    def detect_fields(self, cropped_image: numpy.ndarray) -> HouseholdPrintDetection:
        pass


class HouseholdNextHandwritingDetector(Detector):
    @abstractmethod
    # bgr format
    def detect_fields(
        self, cropped_image: numpy.ndarray
    ) -> HouseholdNextHandwritingDetection:
        pass


class HouseholdNextPrintDetector(Detector):
    @abstractmethod
    # bgr format
    def detect_fields(
        self, cropped_image: numpy.ndarray
    ) -> HouseholdNextPrintDetection:
        pass
