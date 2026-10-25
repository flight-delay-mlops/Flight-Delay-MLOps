"""Request and response shapes of the inference API."""
from pydantic import BaseModel


class Flight(BaseModel):
    """
    One flight of the `instances` array, with the raw BTS column names. (don't change these names)

    Also accepts null and missing key counts as null. (might have to fix later)
    """

    OP_UNIQUE_CARRIER: str | None = None
    TAIL_NUM: str | None = None
    ORIGIN: str | None = None
    ORIGIN_STATE_ABR: str | None = None
    DEST: str | None = None
    DEST_STATE_ABR: str | None = None
    DEP_TIME_BLK: str | None = None
    ARR_TIME_BLK: str | None = None
