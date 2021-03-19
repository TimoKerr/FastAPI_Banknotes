"""
Creation of a BankNote class that inherets from pydantic's BaseModel class
to ensure informative error throws.
"""
# pylint: disable=E0611
from pydantic import BaseModel

# 2. Class which describes Bank Notes measurements
class BankNote(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float
