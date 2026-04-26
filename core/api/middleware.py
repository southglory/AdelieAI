import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from core.logging import get_logger, request_id_var

log = get_logger("differentia.http")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-Id") or uuid.uuid4().hex[:12]
        token = request_id_var.set(rid)
        t0 = time.perf_counter()
        try:
            response: Response = await call_next(request)
        except Exception:
            elapsed = int((time.perf_counter() - t0) * 1000)
            log.exception(
                "request_failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": elapsed,
                },
            )
            request_id_var.reset(token)
            raise

        elapsed = int((time.perf_counter() - t0) * 1000)
        response.headers["X-Request-Id"] = rid
        log.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": elapsed,
            },
        )
        request_id_var.reset(token)
        return response
