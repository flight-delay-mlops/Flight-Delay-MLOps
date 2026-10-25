"""Request and response shapes of the inference API."""
from datetime import date

from pydantic import BaseModel


class Flight(BaseModel):
    """
    One flight of the `instances` array, with the raw BTS column names. (don't change these names)

    Also accepts null and missing key counts as null. (might have to fix later)
    """

    FL_DATE: date | None = None
    OP_UNIQUE_CARRIER: str | None = None
    OP_CARRIER_FL_NUM: int | None = None
    TAIL_NUM: str | None = None
    ORIGIN: str | None = None
    ORIGIN_STATE_ABR: str | None = None
    DEST: str | None = None
    DEST_STATE_ABR: str | None = None
    CRS_DEP_TIME: int | None = None
    DEP_TIME_BLK: str | None = None
    CRS_ARR_TIME: int | None = None
    ARR_TIME_BLK: str | None = None
    CRS_ELAPSED_TIME: float | None = None
    DISTANCE: float | None = None
