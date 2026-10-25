"""Error responses of the inference API."""
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class ModelNotLoadedError(Exception):
    """Raised when a request needs the model and none is loaded."""


class InputRejectedError(Exception):
    """Raised when the model refuses the input it was given; the message says why."""


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


async def handle_input_rejected(request, error: InputRejectedError):
    """Answer 422 with the reason the model gave."""
    return JSONResponse(status_code=422, content=error_body(str(error)))


async def handle_model_not_loaded(request, error: ModelNotLoadedError):
    """Answer 503: the service is up but cannot classify until a model is loaded."""
    return JSONResponse(status_code=503, content=error_body("Model not loaded"))


async def handle_unexpected_error(request, error: Exception):
    """Answer 500 with a generic message, so no internal detail reaches the client."""
    # Starlette re-raises after answering, so the server log still has the traceback
    return JSONResponse(status_code=500, content=error_body("Internal server error"))
