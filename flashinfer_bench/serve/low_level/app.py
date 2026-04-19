"""FastAPI app for the low-level GPU server."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from multiprocessing import shared_memory
from time import perf_counter
from typing import Literal, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ValidationError
from starlette.datastructures import UploadFile

from flashinfer_bench.serve.low_level.errors import InvalidProgramError, TimeoutError
from flashinfer_bench.serve.low_level.executor import (
    ErrorResult,
    ExecuteResult,
    Instruction,
    Program,
    SharedMemoryBlob,
)
from flashinfer_bench.serve.low_level.multipart import build_success_response
from flashinfer_bench.serve.low_level.worker import (
    LowLevelScheduler,
    WorkerExecuteErrorResponse,
    WorkerExecuteSuccessResponse,
)
from flashinfer_bench.utils import list_cuda_devices
from flashinfer_bench.version import __version__

# ---------------------------------------------------------------------------
# HTTP request/response schemas
# ---------------------------------------------------------------------------


class ExecuteRequest(BaseModel):
    """JSON payload stored in the `program` multipart field for `/execute`."""

    instructions: list[Instruction]
    """Ordered instruction sequence to execute."""
    timeout_seconds: float | None = None
    """Optional per-request timeout override in seconds."""

    @property
    def program(self) -> Program:
        """Return the execution program without HTTP-only request metadata.

        Returns
        -------
        Program
            Program object containing only execution instructions.
        """
        return Program(instructions=self.instructions)


class WorkerStatus(BaseModel):
    """Health summary for a single GPU worker."""

    gpu_id: int
    """Physical GPU index assigned to the worker."""
    status: Literal["idle", "busy", "restarting"]
    """Current worker runtime status."""
    queue_length: int
    """Number of in-flight requests assigned to this worker."""
    uptime_seconds: int
    """Worker uptime since the last process start."""


class HealthResponse(BaseModel):
    """Response payload for the `/health` endpoint."""

    status: Literal["ok"] = "ok"
    """Health check status."""
    gpu_count: int
    """Number of configured GPU workers."""
    workers: list[WorkerStatus]
    """Per-worker runtime status entries."""


class RequestSharedMemoryStore:
    """Own shared-memory blobs created for one HTTP request."""

    blobs: dict[str, SharedMemoryBlob]
    """Shared-memory blob metadata keyed by blob hash."""
    _shared_memories: list[shared_memory.SharedMemory]
    """Owned shared-memory objects that must be released after the request."""

    def __init__(self) -> None:
        """Initialize an empty request-local shared-memory store.

        Returns
        -------
        None
            This constructor initializes internal storage in place.
        """
        self.blobs: dict[str, SharedMemoryBlob] = {}
        self._shared_memories: list[shared_memory.SharedMemory] = []

    def add_blob(self, blob_hash: str, blob_bytes: bytes) -> None:
        """Create one shared-memory blob and record its metadata.

        Parameters
        ----------
        blob_hash
            Blob hash used as the logical blob key.
        blob_bytes
            Raw blob payload received from the HTTP request.

        Returns
        -------
        None
            This method mutates the shared-memory store in place.
        """
        if blob_hash in self.blobs:
            raise InvalidProgramError(f"Duplicate blob: {blob_hash}")

        shared_memory_size = max(1, len(blob_bytes))
        blob_shared_memory = shared_memory.SharedMemory(create=True, size=shared_memory_size)
        try:
            blob_shared_memory.buf[: len(blob_bytes)] = blob_bytes
        except Exception:
            blob_shared_memory.close()
            blob_shared_memory.unlink()
            raise

        self._shared_memories.append(blob_shared_memory)
        self.blobs[blob_hash] = SharedMemoryBlob(name=blob_shared_memory.name, size=len(blob_bytes))

    def cleanup(self) -> None:
        """Release all request shared-memory blobs.

        Returns
        -------
        None
            This method releases owned shared-memory objects in place.
        """
        for blob_shared_memory in self._shared_memories:
            try:
                blob_shared_memory.close()
            except Exception:
                pass
            try:
                blob_shared_memory.unlink()
            except FileNotFoundError:
                pass
        self._shared_memories.clear()
        self.blobs.clear()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

_scheduler: Optional[LowLevelScheduler] = None


def _get_scheduler() -> LowLevelScheduler:
    """Return the initialized scheduler instance.

    Returns
    -------
    LowLevelScheduler
        Initialized scheduler singleton for the current process.
    """
    if _scheduler is None:
        raise RuntimeError("Low-level server is not initialized")
    return _scheduler


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Own the scheduler lifecycle for the module-level FastAPI app.

    Parameters
    ----------
    app
        FastAPI application bound to this lifespan handler.

    Yields
    ------
    None
        Control returns to FastAPI while the app is serving requests.
    """
    yield
    if _scheduler is not None:
        _scheduler.close()


app = FastAPI(title="FlashInfer-Bench Low-Level Server", version=__version__, lifespan=_lifespan)


def create_app(devices: list[str], default_timeout_seconds: int = 30) -> FastAPI:
    """Create the low-level app with a concrete scheduler.

    Parameters
    ----------
    devices
        CUDA device strings assigned to worker processes.
    default_timeout_seconds
        Default request timeout in seconds.

    Returns
    -------
    FastAPI
        Configured FastAPI application instance.
    """
    global _scheduler
    _scheduler = LowLevelScheduler(devices=devices, default_timeout_seconds=default_timeout_seconds)
    return app


def create_default_app() -> FastAPI:
    """Factory for ad-hoc uvicorn startup.

    Returns
    -------
    FastAPI
        Configured FastAPI application using detected CUDA devices.
    """
    devices = list_cuda_devices()
    return create_app(devices=devices, default_timeout_seconds=30)


@app.post(
    "/execute",
    responses={
        200: {"description": "Success (multipart)", "model": ExecuteResult},
        400: {"description": "Invalid program or execution failed", "model": ErrorResult},
        408: {"description": "Timeout", "model": ErrorResult},
    },
)
async def execute(request: Request):
    """Execute a program on GPU.

    Input: multipart/form-data
      - program: ExecuteRequest JSON
      - blob:<sha256>: binary data (referenced by instructions)
    Output: multipart/form-data
      - result: ExecuteResult JSON (including optional stdout/stderr)
      - return:<key>: binary return values (tensors, bytes)

    Parameters
    ----------
    request
        Incoming HTTP request carrying one multipart execution payload.

    Returns
    -------
    Response
        Multipart success response or JSON error response.
    """
    execute_start_time = perf_counter()
    blob_store = RequestSharedMemoryStore()
    validate_program_ms: float | None = None
    scheduler_execute_ms: float | None = None
    build_response_ms: float | None = None
    request_cleanup_ms: float | None = None
    status_code: int | None = None
    try:
        raw_json = await _parse_request_into_shared_memory(request, blob_store)
        try:
            validate_start_time = perf_counter()
            execute_request = ExecuteRequest.model_validate(raw_json)
            validate_program_ms = (perf_counter() - validate_start_time) * 1000.0
        except ValidationError as error:
            raise InvalidProgramError(str(error)) from error
        program = execute_request.program
        scheduler_execute_start_time = perf_counter()
        response = await asyncio.to_thread(
            _get_scheduler().execute, program, execute_request.timeout_seconds, blob_store.blobs
        )
        scheduler_execute_ms = (perf_counter() - scheduler_execute_start_time) * 1000.0
        if isinstance(response, WorkerExecuteErrorResponse):
            error_result: ErrorResult = response.error
            status_code = 400
            return JSONResponse(
                status_code=status_code, content=error_result.model_dump(exclude_none=True)
            )

        success_response: WorkerExecuteSuccessResponse = response
        result: ExecuteResult = success_response.result
        binary_parts: dict[str, bytes] = success_response.binary_parts
        build_response_start_time = perf_counter()
        body, content_type = build_success_response(
            result_payload=result.model_dump(), binary_parts=binary_parts
        )
        build_response_ms = (perf_counter() - build_response_start_time) * 1000.0
        status_code = 200
        return Response(content=body, media_type=content_type)
    except InvalidProgramError as error:
        result = ErrorResult(error="invalid_program", message=error.message)
        status_code = 400
        return JSONResponse(status_code=status_code, content=result.model_dump(exclude_none=True))
    except TimeoutError as error:
        result = ErrorResult(error="timeout", timeout_seconds=error.timeout_seconds)
        status_code = 408
        return JSONResponse(status_code=status_code, content=result.model_dump(exclude_none=True))
    finally:
        cleanup_start_time = perf_counter()
        blob_store.cleanup()
        request_cleanup_ms = (perf_counter() - cleanup_start_time) * 1000.0
        print(
            "APP_TIMING "
            f"status_code={status_code} "
            f"total_execute_ms={(perf_counter() - execute_start_time) * 1000.0:.3f} "
            f"validate_program_ms={validate_program_ms} "
            f"scheduler_execute_ms={scheduler_execute_ms} "
            f"build_response_ms={build_response_ms} "
            f"request_cleanup_ms={request_cleanup_ms}",
            flush=True,
        )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Server health check with per-worker status.

    Returns
    -------
    HealthResponse
        Structured health summary for all managed workers.
    """
    scheduler = _get_scheduler()
    workers = [
        WorkerStatus(
            gpu_id=worker.gpu_id,
            status=worker.status,
            queue_length=worker.queue_length,
            uptime_seconds=worker.uptime_seconds,
        )
        for worker in scheduler._workers
    ]
    return HealthResponse(gpu_count=len(workers), workers=workers)


async def _parse_request_into_shared_memory(
    request: Request, blob_store: RequestSharedMemoryStore
) -> dict:
    """Parse multipart request parts and materialize uploaded blobs into shared memory.

    Parameters
    ----------
    request
        Incoming multipart HTTP request.
    blob_store
        Request-local shared-memory store that will receive uploaded blobs.

    Returns
    -------
    dict
        Raw decoded JSON object from the `program` multipart field.
    """
    parse_start_time = perf_counter()
    form_parse_start_time = perf_counter()
    form = await request.form()
    request_form_ms = (perf_counter() - form_parse_start_time) * 1000.0
    program: Optional[dict] = None
    num_blobs = 0
    total_blob_bytes = 0
    program_read_ms = 0.0
    program_json_decode_ms = 0.0
    blob_read_ms = 0.0
    blob_store_ms = 0.0

    for key, value in form.multi_items():
        if key == "program":
            program_read_start_time = perf_counter()
            program_bytes = await _read_form_part_bytes(value)
            program_read_ms += (perf_counter() - program_read_start_time) * 1000.0
            try:
                program_json_decode_start_time = perf_counter()
                program = json.loads(program_bytes.decode("utf-8"))
                program_json_decode_ms += (perf_counter() - program_json_decode_start_time) * 1000.0
            except Exception as error:
                raise InvalidProgramError(f"Invalid program JSON: {error}") from error
            continue

        if key.startswith("blob:"):
            blob_hash = key.split(":", 1)[1]
            blob_read_start_time = perf_counter()
            blob_bytes = await _read_form_part_bytes(value)
            blob_read_ms += (perf_counter() - blob_read_start_time) * 1000.0
            blob_store_start_time = perf_counter()
            blob_store.add_blob(blob_hash, blob_bytes)
            blob_store_ms += (perf_counter() - blob_store_start_time) * 1000.0
            num_blobs += 1
            total_blob_bytes += len(blob_bytes)

    if program is None:
        raise InvalidProgramError("Missing program part")
    print(
        "APP_PARSE_TIMING "
        f"total_parse_ms={(perf_counter() - parse_start_time) * 1000.0:.3f} "
        f"request_form_ms={request_form_ms:.3f} "
        f"program_read_ms={program_read_ms:.3f} "
        f"program_json_decode_ms={program_json_decode_ms:.3f} "
        f"blob_read_ms={blob_read_ms:.3f} "
        f"blob_store_ms={blob_store_ms:.3f} "
        f"num_blobs={num_blobs} "
        f"total_blob_bytes={total_blob_bytes}",
        flush=True,
    )
    return program


async def _read_form_part_bytes(value: object) -> bytes:
    """Normalize one multipart field into raw bytes.

    Parameters
    ----------
    value
        Multipart field value returned by Starlette form parsing.

    Returns
    -------
    bytes
        Raw bytes representing the multipart field payload.
    """
    if isinstance(value, UploadFile):
        return await value.read()
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, bytes):
        return value
    raise InvalidProgramError(f"Unsupported multipart part type: {type(value)!r}")


def main():
    """Run the low-level server as a standalone uvicorn process.

    Returns
    -------
    None
        This function configures logging, creates the app, and starts Uvicorn.
    """
    import argparse
    import logging

    import uvicorn

    parser = argparse.ArgumentParser(description="Low-level GPU judge server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument(
        "--devices", type=str, default=None, help="Comma-separated, e.g. cuda:0,cuda:1"
    )
    parser.add_argument(
        "--timeout", type=int, default=30, help="Default execution timeout in seconds"
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    devices = args.devices.split(",") if args.devices else list_cuda_devices()
    if not devices:
        raise RuntimeError("No CUDA devices available")

    create_app(devices=devices, default_timeout_seconds=args.timeout)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
