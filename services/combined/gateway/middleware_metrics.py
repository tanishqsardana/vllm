from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class GatewayMetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        metrics = getattr(request.app.state, "metrics", None)
        if metrics is None:
            return await call_next(request)

        start = time.perf_counter()
        status = 500
        metrics.inc_inflight()

        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            if route is not None and getattr(route, "path", None):
                route_label = route.path
            else:
                route_label = "unmatched"

            duration_seconds = max(0.0, time.perf_counter() - start)
            metrics.observe_http_duration(route=route_label, method=request.method, duration_seconds=duration_seconds)
            metrics.inc_http_requests(route=route_label, method=request.method, status=status)
            metrics.dec_inflight()

