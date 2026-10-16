"""Shared enums/types used across request and response schemas."""
from enum import Enum


class Season(str, Enum):
    winter = "winter"
    pre_monsoon = "pre_monsoon"
    monsoon = "monsoon"
    post_monsoon = "post_monsoon"
