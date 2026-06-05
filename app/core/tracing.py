from __future__ import annotations

import logging
from collections.abc import Mapping

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span

from app.core.config import get_settings

logger = logging.getLogger(__name__)

SafeSpanAttribute = str | bool | int | float

_tracing_initialized = False
_libraries_instrumented = False


def setup_tracing(app: FastAPI) -> None:
    settings = get_settings()
    if not settings.TRACING_ENABLED:
        return

    _setup_tracer_provider()
    _instrument_libraries()
    _instrument_fastapi(app)


def set_span_attributes(
    span: Span,
    attributes: Mapping[str, SafeSpanAttribute | None],
) -> None:
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)


def _setup_tracer_provider() -> None:
    global _tracing_initialized

    if _tracing_initialized:
        return

    settings = get_settings()
    resource = Resource.create(
        {
            "service.name": settings.OTEL_SERVICE_NAME,
            "deployment.environment": settings.OTEL_ENVIRONMENT,
        }
    )
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracing_initialized = True


def _instrument_libraries() -> None:
    global _libraries_instrumented

    if _libraries_instrumented:
        return

    try:
        AsyncPGInstrumentor().instrument()
        HTTPXClientInstrumentor().instrument()
        LoggingInstrumentor().instrument(set_logging_format=False)
        _libraries_instrumented = True
    except Exception:
        logger.exception("Failed to configure OpenTelemetry auto-instrumentation")


def _instrument_fastapi(app: FastAPI) -> None:
    if getattr(app.state, "tracing_instrumented", False):
        return

    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=trace.get_tracer_provider(),
    )
    app.state.tracing_instrumented = True
