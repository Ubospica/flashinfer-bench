"""Program executor for the low-level GPU server."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Literal, Tuple, Union

import numpy as np
import torch
import tvm_ffi
from pydantic import BaseModel, ValidationError
from tvm_ffi.registry import list_global_func_names

from flashinfer_bench.serve.low_level.errors import ExecutionFailedError, InvalidProgramError
from flashinfer_bench.utils import dtype_str_to_torch_dtype

# ---------------------------------------------------------------------------
# Instruction schemas
# ---------------------------------------------------------------------------

SUPPORTED_DTYPES = Literal[
    "float16", "bfloat16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "bool"
]


class RegisterRef(BaseModel):
    """Reference a value previously stored in the register file."""

    r: int
    """Register index to read from."""


Operand = Union[RegisterRef, int, float, str, bool, list, None]


class UploadPythonModuleInstruction(BaseModel):
    """Import a Python module blob into the worker process."""

    op: Literal["upload_python_module"]
    """Instruction opcode."""
    blob: str
    """Blob key referencing the uploaded Python module source."""


class UploadFfiModuleInstruction(BaseModel):
    """Load a TVM FFI shared library blob and store its module handle."""

    op: Literal["upload_ffi_module"]
    """Instruction opcode."""
    blob: str
    """Blob key referencing the uploaded shared library bytes."""
    dst: int
    """Destination register for the loaded FFI module handle."""


class UploadTensorInstruction(BaseModel):
    """Load a tensor blob onto the worker GPU and store it in a register."""

    op: Literal["upload_tensor"]
    """Instruction opcode."""
    dst: int
    """Destination register for the uploaded tensor."""
    blob: str
    """Blob key referencing the raw tensor bytes."""
    shape: list[int]
    """Tensor shape in row-major order."""
    dtype: SUPPORTED_DTYPES
    """Tensor dtype name."""


class CallInstruction(BaseModel):
    """Call a function with literal operands or register references."""

    op: Literal["call"]
    """Instruction opcode."""
    func: str
    """Function name to invoke."""
    args: list[Operand]
    """Function arguments in call order."""
    dst: int | None = None
    """Destination register for the return value."""
    module: RegisterRef | None = None
    """Optional register reference to a module handle used for function lookup."""


class ReturnInstruction(BaseModel):
    """Expose a register value in the final response payload."""

    op: Literal["return"]
    """Instruction opcode."""
    reg: int
    """Register index to export."""
    key: str
    """Response key used in the returns map."""


Instruction = Union[
    UploadPythonModuleInstruction,
    UploadFfiModuleInstruction,
    UploadTensorInstruction,
    CallInstruction,
    ReturnInstruction,
]


class Program(BaseModel):
    """Executable low-level program independent of HTTP request metadata."""

    instructions: list[Instruction]
    """Ordered instruction sequence to execute."""


def parse_program(raw: dict) -> Program:
    """Parse and validate a raw JSON dict into a typed `Program`.

    Parameters
    ----------
    raw
        Raw JSON object decoded from the request payload.

    Returns
    -------
    Program
        Validated execution program.
    """
    try:
        return Program.model_validate(raw)
    except ValidationError as error:
        raise InvalidProgramError(str(error)) from error


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class TensorReturnValue(BaseModel):
    """Tensor return value described by metadata plus a binary blob part."""

    type: Literal["tensor"] = "tensor"
    """Return value kind."""
    dtype: SUPPORTED_DTYPES
    """Tensor dtype name."""
    shape: list[int]
    """Tensor shape in row-major order."""
    blob: str
    """Multipart part name containing the raw tensor bytes."""


class BytesReturnValue(BaseModel):
    """Opaque bytes return value stored in a multipart blob part."""

    type: Literal["bytes"] = "bytes"
    """Return value kind."""
    blob: str
    """Multipart part name containing the raw bytes."""


class JsonReturnValue(BaseModel):
    """JSON-serializable return value embedded directly in the result payload."""

    type: Literal["json"] = "json"
    """Return value kind."""
    value: Any
    """JSON-serializable return value."""


ReturnValue = Union[TensorReturnValue, BytesReturnValue, JsonReturnValue]


class ExecuteResult(BaseModel):
    """Successful execution result returned in the `result` multipart field."""

    status: Literal["ok"] = "ok"
    """Execution status."""
    returns: dict[str, ReturnValue]
    """Named return values exported by `return` instructions."""
    elapsed_ms: float
    """Wall-clock execution time in milliseconds."""
    stdout: str | None = None
    """Captured standard output from the worker process."""
    stderr: str | None = None
    """Captured standard error from the worker process."""


class ErrorResult(BaseModel):
    """Structured execution error returned as JSON when the request fails."""

    status: Literal["error"] = "error"
    """Execution status."""
    error: str
    """Stable machine-readable error code."""
    message: str | None = None
    """Human-readable error summary."""
    instruction_index: int | None = None
    """Zero-based instruction index associated with the failure, when available."""
    timeout_seconds: float | None = None
    """Effective timeout that was exceeded, when the error is a timeout."""
    stdout: str | None = None
    """Captured standard output before the failure."""
    stderr: str | None = None
    """Captured standard error before the failure."""


class SharedMemoryBlob(BaseModel):
    """Shared-memory blob metadata passed from the frontend to a worker."""

    name: str
    """Operating-system shared-memory object name."""
    size: int
    """Logical blob size in bytes."""


# ---------------------------------------------------------------------------
# Dtype mappings
# ---------------------------------------------------------------------------

_DTYPE_TO_NUMPY: Dict[str, np.dtype] = {
    "float16": np.dtype(np.float16),
    "bfloat16": np.dtype(np.uint16),
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
    "int8": np.dtype(np.int8),
    "int16": np.dtype(np.int16),
    "int32": np.dtype(np.int32),
    "int64": np.dtype(np.int64),
    "uint8": np.dtype(np.uint8),
    "bool": np.dtype(np.bool_),
}

_TORCH_DTYPE_TO_NAME: Dict[torch.dtype, str] = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.bool: "bool",
}

# ---------------------------------------------------------------------------
# Request context
# ---------------------------------------------------------------------------


@dataclass
class RequestContext:
    register_file: Dict[int, Any] = field(default_factory=dict)
    """Request-local register file keyed by register index."""
    registered_function_names: set[str] = field(default_factory=set)
    """Global function names registered during the current request."""
    loaded_shared_modules: list[Any] = field(default_factory=list)
    """Uploaded FFI module handles kept alive for the current request."""
    loaded_python_modules: list[Any] = field(default_factory=list)
    """Imported Python module objects kept alive for the current request."""
    loaded_python_module_names: list[str] = field(default_factory=list)
    """Temporary module names inserted into `sys.modules` for uploaded Python modules."""
    blobs: Dict[str, SharedMemoryBlob] = field(default_factory=dict)
    """Uploaded request blobs addressable by blob hash."""
    attached_shared_memories: Dict[str, shared_memory.SharedMemory] = field(default_factory=dict)
    """Worker-local shared-memory attachments opened for uploaded blobs."""

    def get_register(self, register_index: int) -> Any:
        """Read a value from the request register file.

        Parameters
        ----------
        register_index
            Register index to read.

        Returns
        -------
        Any
            Value currently stored in the register.
        """
        if register_index not in self.register_file:
            raise InvalidProgramError(f"Register r{register_index} is not defined")
        return self.register_file[register_index]

    def set_register(self, register_index: int, value: Any) -> None:
        """Store a value in the request register file.

        Parameters
        ----------
        register_index
            Register index to write.
        value
            Value to store in the register.

        Returns
        -------
        None
            This method updates the request context in place.
        """
        self.register_file[register_index] = value

    def is_loaded_shared_module(self, value: Any) -> bool:
        """Check whether a value is a request-local uploaded FFI module handle.

        Parameters
        ----------
        value
            Candidate value to test.

        Returns
        -------
        bool
            Whether the value matches one uploaded FFI module handle by identity.
        """
        return any(value is shared_module for shared_module in self.loaded_shared_modules)

    def get_blob_buffer(self, blob_hash: str) -> memoryview:
        """Return a memoryview over one uploaded blob.

        Parameters
        ----------
        blob_hash
            Blob hash referenced by the program instruction.

        Returns
        -------
        memoryview
            Shared-memory view spanning the logical blob bytes.
        """
        if blob_hash not in self.blobs:
            raise InvalidProgramError(f"Missing blob: {blob_hash}")
        blob = self.blobs[blob_hash]
        if blob_hash not in self.attached_shared_memories:
            try:
                self.attached_shared_memories[blob_hash] = shared_memory.SharedMemory(
                    name=blob.name
                )
            except FileNotFoundError as error:
                raise InvalidProgramError(f"Missing shared-memory blob: {blob_hash}") from error
        return self.attached_shared_memories[blob_hash].buf[: blob.size]

    def read_blob_bytes(self, blob_hash: str) -> bytes:
        """Copy one uploaded blob into an immutable bytes object.

        Parameters
        ----------
        blob_hash
            Blob hash referenced by the program instruction.

        Returns
        -------
        bytes
            Immutable copy of the requested blob payload.
        """
        blob_buffer = self.get_blob_buffer(blob_hash)
        try:
            return bytes(blob_buffer)
        finally:
            blob_buffer.release()

    def cleanup(self) -> None:
        """Release request-local state after execution finishes.

        Returns
        -------
        None
            This method releases request-owned resources in place.
        """
        self.register_file.clear()
        for function_name in sorted(self.registered_function_names):
            try:
                tvm_ffi.remove_global_func(function_name)
            except Exception:
                pass
        self.loaded_shared_modules.clear()
        for module_name in self.loaded_python_module_names:
            sys.modules.pop(module_name, None)
        self.loaded_python_modules.clear()
        self.loaded_python_module_names.clear()
        for attached_shared_memory in self.attached_shared_memories.values():
            attached_shared_memory.close()
        self.attached_shared_memories.clear()
        self.blobs.clear()
        self.registered_function_names.clear()


# ---------------------------------------------------------------------------
# Program execution
# ---------------------------------------------------------------------------


def execute_program(
    program: Program, blobs: Dict[str, SharedMemoryBlob]
) -> Tuple[ExecuteResult, Dict[str, bytes]]:
    """Execute a validated program against request blobs.

    Parameters
    ----------
    program
        Validated low-level program to execute.
    blobs
        Uploaded request blobs stored in shared memory and keyed by blob hash.

    Returns
    -------
    tuple[ExecuteResult, dict[str, bytes]]
        Structured result payload plus binary multipart parts.
    """
    context = RequestContext(blobs=blobs)
    returned_values: Dict[str, Any] = {}
    execute_program_start_time = perf_counter()
    instruction_loop_ms: float | None = None
    serialize_returns_ms: float | None = None
    elapsed_ms: float | None = None
    cleanup_ms: float | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="fib_low_level_worker_") as temporary_directory:
            tmp_dir = Path(temporary_directory)
            start_time = perf_counter()
            instruction_loop_start_time = perf_counter()
            for instruction_index, instruction in enumerate(program.instructions):
                _execute_instruction(
                    context, tmp_dir, instruction, instruction_index, returned_values
                )
            instruction_loop_ms = (perf_counter() - instruction_loop_start_time) * 1000.0
            serialize_returns_start_time = perf_counter()
            returns, binary_parts = _serialize_returns(returned_values)
            serialize_returns_ms = (perf_counter() - serialize_returns_start_time) * 1000.0
            elapsed_ms = (perf_counter() - start_time) * 1000.0
            result = ExecuteResult(returns=returns, elapsed_ms=elapsed_ms)
            return result, binary_parts
    finally:
        cleanup_start_time = perf_counter()
        context.cleanup()
        cleanup_ms = (perf_counter() - cleanup_start_time) * 1000.0
        print(
            "EXECUTOR_TIMING "
            f"total_execute_program_ms={(perf_counter() - execute_program_start_time) * 1000.0:.3f} "
            f"elapsed_ms={elapsed_ms} "
            f"instruction_loop_ms={instruction_loop_ms} "
            f"serialize_returns_ms={serialize_returns_ms} "
            f"cleanup_ms={cleanup_ms}",
            flush=True,
        )


def _execute_instruction(
    context: RequestContext,
    tmp_dir: Path,
    instruction: Instruction,
    instruction_index: int,
    returned_values: Dict[str, Any],
) -> None:
    """Dispatch one typed instruction to the matching execution helper.

    Parameters
    ----------
    context
        Request-local execution state.
    tmp_dir
        Temporary directory used for request-local module files.
    instruction
        Typed instruction to execute.
    instruction_index
        Zero-based instruction index in the program.
    returned_values
        Mutable map populated by `return` instructions.

    Returns
    -------
    None
        This function mutates the request context and return map in place.
    """
    if isinstance(instruction, UploadPythonModuleInstruction):
        _execute_upload_python_module(context, tmp_dir, instruction, instruction_index)
    elif isinstance(instruction, UploadFfiModuleInstruction):
        _execute_upload_ffi_module(context, tmp_dir, instruction, instruction_index)
    elif isinstance(instruction, UploadTensorInstruction):
        _execute_upload_tensor(context, instruction)
    elif isinstance(instruction, CallInstruction):
        _execute_call(context, instruction, instruction_index)
    elif isinstance(instruction, ReturnInstruction):
        _execute_return(context, instruction, returned_values)


def _execute_upload_python_module(
    context: RequestContext,
    tmp_dir: Path,
    instruction: UploadPythonModuleInstruction,
    instruction_index: int,
) -> None:
    """Import a Python module blob and track its registered global functions.

    Parameters
    ----------
    context
        Request-local execution state.
    tmp_dir
        Temporary directory used for request-local module files.
    instruction
        Python module upload instruction to execute.
    instruction_index
        Zero-based instruction index in the program.

    Returns
    -------
    None
        This function mutates the request context in place.
    """
    try:
        blob_bytes = context.read_blob_bytes(instruction.blob)
        _load_python_module_blob(context, tmp_dir, instruction.blob, blob_bytes)
    except (InvalidProgramError, ExecutionFailedError):
        raise
    except Exception as error:
        raise ExecutionFailedError(
            message=str(error), instruction_index=instruction_index
        ) from error


def _execute_upload_ffi_module(
    context: RequestContext,
    tmp_dir: Path,
    instruction: UploadFfiModuleInstruction,
    instruction_index: int,
) -> None:
    """Load a shared library blob and store the resulting module handle.

    Parameters
    ----------
    context
        Request-local execution state.
    tmp_dir
        Temporary directory used for request-local module files.
    instruction
        FFI module upload instruction to execute.
    instruction_index
        Zero-based instruction index in the program.

    Returns
    -------
    None
        This function mutates the request context in place.
    """
    try:
        blob_bytes = context.read_blob_bytes(instruction.blob)
        loaded_module = _load_shared_library_blob(context, tmp_dir, instruction.blob, blob_bytes)
        context.set_register(instruction.dst, loaded_module)
    except (InvalidProgramError, ExecutionFailedError):
        raise
    except Exception as error:
        raise ExecutionFailedError(
            message=str(error), instruction_index=instruction_index
        ) from error


def _load_python_module_blob(
    context: RequestContext, tmp_dir: Path, blob_hash: str, blob_bytes: bytes
) -> None:
    """Import a Python module blob and record newly registered global functions.

    Parameters
    ----------
    context
        Request-local execution state.
    tmp_dir
        Temporary directory used for request-local module files.
    blob_hash
        Blob hash used to derive the temporary file name.
    blob_bytes
        Python module source bytes.

    Returns
    -------
    None
        This function mutates the request context in place.
    """
    before_function_names = set(list_global_func_names())
    module_name = f"flashinfer_bench_uploaded_{uuid.uuid4().hex}"
    source_path = tmp_dir / f"{blob_hash}.py"
    source_path.write_bytes(blob_bytes)
    spec = importlib.util.spec_from_file_location(module_name, str(source_path))
    if spec is None or spec.loader is None:
        raise InvalidProgramError(f"Failed to build import spec for blob: {blob_hash}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        after_function_names = set(list_global_func_names())
        context.registered_function_names.update(after_function_names - before_function_names)
        sys.modules.pop(module_name, None)
        raise

    after_function_names = set(list_global_func_names())
    context.registered_function_names.update(after_function_names - before_function_names)
    context.loaded_python_modules.append(module)
    context.loaded_python_module_names.append(module_name)


def _load_shared_library_blob(
    context: RequestContext, tmp_dir: Path, blob_hash: str, blob_bytes: bytes
):
    """Load a shared library blob and record newly registered global functions.

    Parameters
    ----------
    context
        Request-local execution state.
    tmp_dir
        Temporary directory used for request-local module files.
    blob_hash
        Blob hash used to derive the temporary file name.
    blob_bytes
        Shared-library bytes.

    Returns
    -------
    Any
        Loaded FFI module handle returned by `tvm_ffi.load_module`.
    """
    if not blob_bytes.startswith(b"\x7fELF"):
        raise InvalidProgramError(
            f"upload_ffi_module expects an ELF shared library blob: {blob_hash}"
        )

    before_function_names = set(list_global_func_names())
    shared_library_path = tmp_dir / f"{blob_hash}.so"
    shared_library_path.write_bytes(blob_bytes)
    try:
        loaded_module = tvm_ffi.load_module(str(shared_library_path), keep_module_alive=False)
    except Exception:
        after_function_names = set(list_global_func_names())
        context.registered_function_names.update(after_function_names - before_function_names)
        raise

    after_function_names = set(list_global_func_names())
    context.registered_function_names.update(after_function_names - before_function_names)
    context.loaded_shared_modules.append(loaded_module)
    return loaded_module


def _execute_upload_tensor(context: RequestContext, instruction: UploadTensorInstruction) -> None:
    """Materialize a tensor blob on the worker GPU and store it in a register.

    Parameters
    ----------
    context
        Request-local execution state.
    instruction
        Tensor upload instruction to execute.

    Returns
    -------
    None
        This function mutates the request context in place.
    """
    blob_buffer = context.get_blob_buffer(instruction.blob)
    numpy_dtype = _DTYPE_TO_NUMPY[instruction.dtype]
    expected_nbytes = int(np.prod(instruction.shape, dtype=np.int64)) * numpy_dtype.itemsize
    try:
        if len(blob_buffer) != expected_nbytes:
            raise InvalidProgramError(
                f"upload_tensor blob size mismatch: expected {expected_nbytes} bytes, got {len(blob_buffer)}"
            )

        numpy_array = (
            np.frombuffer(blob_buffer, dtype=numpy_dtype).copy().reshape(instruction.shape)
        )
        torch_dtype = dtype_str_to_torch_dtype(instruction.dtype)
        if instruction.dtype == "bfloat16":
            tensor = torch.from_numpy(numpy_array.view(np.int16)).view(torch.bfloat16)
        else:
            tensor = torch.from_numpy(numpy_array).to(dtype=torch_dtype)
        tensor = tensor.to("cuda:0")
        context.set_register(instruction.dst, tensor)
    finally:
        blob_buffer.release()


def _execute_call(
    context: RequestContext, instruction: CallInstruction, instruction_index: int
) -> None:
    """Invoke a function and optionally write the return value to a register.

    Parameters
    ----------
    context
        Request-local execution state.
    instruction
        Function call instruction to execute.
    instruction_index
        Zero-based instruction index in the program.

    Returns
    -------
    None
        This function mutates the request context in place when `dst` is provided.
    """
    try:
        resolved_args = [_resolve_operand(context, item) for item in instruction.args]
        function = _resolve_function(context, instruction.func, instruction.module)
        if function is None:
            raise ExecutionFailedError(
                message=f"Unknown function: {instruction.func}", instruction_index=instruction_index
            )
        result = function(*resolved_args)
        if instruction.dst is not None:
            context.set_register(instruction.dst, result)
    except (InvalidProgramError, ExecutionFailedError):
        raise
    except Exception as error:
        raise ExecutionFailedError(
            message=str(error), instruction_index=instruction_index
        ) from error


def _resolve_function(context: RequestContext, function_name: str, module_ref: RegisterRef | None):
    """Resolve a callable either from global scope or from a module register.

    Parameters
    ----------
    context
        Request-local execution state.
    function_name
        Function name to resolve.
    module_ref
        Optional register reference pointing to an uploaded FFI module handle.

    Returns
    -------
    Any
        Resolved callable object, or `None` when the global function is missing.
    """
    if module_ref is None:
        return tvm_ffi.get_global_func(function_name, allow_missing=True)

    module_value = context.get_register(module_ref.r)
    if context.is_loaded_shared_module(module_value):
        return module_value.get_function(function_name)
    raise InvalidProgramError("call.module must resolve to an uploaded FFI module handle")


def _execute_return(
    context: RequestContext, instruction: ReturnInstruction, returned_values: Dict[str, Any]
) -> None:
    """Expose a register value in the final named return map.

    Parameters
    ----------
    context
        Request-local execution state.
    instruction
        Return instruction to execute.
    returned_values
        Mutable map populated with named return values.

    Returns
    -------
    None
        This function mutates the return map in place.
    """
    returned_values[instruction.key] = context.get_register(instruction.reg)


def _resolve_operand(context: RequestContext, operand: Any) -> Any:
    """Resolve a literal operand or a register reference.

    Parameters
    ----------
    context
        Request-local execution state.
    operand
        Literal value or register reference appearing in the instruction.

    Returns
    -------
    Any
        Resolved Python value ready to pass into the target function.
    """
    if isinstance(operand, RegisterRef):
        return context.get_register(operand.r)
    return operand


def _serialize_returns(
    returned_values: Dict[str, Any],
) -> Tuple[Dict[str, ReturnValue], Dict[str, bytes]]:
    """Convert returned register values into the HTTP result payload.

    Parameters
    ----------
    returned_values
        Named values exported by `return` instructions.

    Returns
    -------
    tuple[dict[str, ReturnValue], dict[str, bytes]]
        Serialized return metadata plus binary multipart parts.
    """
    returns: Dict[str, ReturnValue] = {}
    binary_parts: Dict[str, bytes] = {}

    for key, value in returned_values.items():
        if isinstance(value, torch.Tensor):
            tensor = value.detach().contiguous().cpu()
            part_name = f"return:{key}"
            binary_parts[part_name] = tensor.numpy().tobytes()
            returns[key] = TensorReturnValue(
                dtype=_TORCH_DTYPE_TO_NAME[tensor.dtype], shape=list(tensor.shape), blob=part_name
            )
        elif isinstance(value, (bytes, bytearray)):
            part_name = f"return:{key}"
            binary_parts[part_name] = bytes(value)
            returns[key] = BytesReturnValue(blob=part_name)
        else:
            returns[key] = JsonReturnValue(value=value)

    return returns, binary_parts
