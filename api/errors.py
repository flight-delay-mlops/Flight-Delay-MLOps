"""Error responses of the inference API."""
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def error_body(detail):
    """Body of every error response: change the format of all of them here."""
    return {"detail": detail}


async def handle_validation_error(request, error: RequestValidationError):
    """Answer 422, naming where each problem is and what it is, without echoing the input."""
    issues = [
        {"loc": list(issue["loc"]), "msg": issue["msg"], "type": issue["type"]}
        for issue in error.errors()
    ]
    return JSONResponse(status_code=422, content=error_body(issues))
