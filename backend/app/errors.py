"""C5 error format: every error is {"error": {"code", "message"}}."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.schemas import ApiError as ApiErrorBody
from app.schemas import ErrorCode, ErrorDetail


class ApiError(Exception):
    def __init__(self, status: int, code: ErrorCode, message: str):
        self.status, self.code, self.message = status, code, message


def _body(status: int, code: ErrorCode, message: str) -> JSONResponse:
    detail = ErrorDetail(code=code, message=message)
    return JSONResponse(ApiErrorBody(error=detail).model_dump(), status_code=status)


def install(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error(_: Request, e: ApiError):
        return _body(e.status, e.code, e.message)

    @app.exception_handler(RequestValidationError)
    async def bad_request(_: Request, e: RequestValidationError):
        fields = ", ".join(".".join(str(p) for p in err["loc"][1:]) for err in e.errors())
        return _body(400, "bad_request", f"Missing or malformed field: {fields or 'request'}.")

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, e: HTTPException):
        code = "model_not_found" if e.status_code == 404 else "bad_request"
        return _body(e.status_code, code, str(e.detail))

    @app.exception_handler(Exception)
    async def internal(_: Request, e: Exception):
        return _body(500, "internal_error", "Something went wrong on the server.")
