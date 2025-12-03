import atexit
import signal
import threading
import uuid
from typing import Any, Dict, Iterable, List, Optional

import torch

from flashinfer_bench.data import (
    Definition,
    InputSpec,
    RandomInput,
    SafetensorsInput,
    Trace,
    TraceSet,
    Workload,
)
from flashinfer_bench.env import get_fib_dataset_path, get_fib_enable_tracing
from flashinfer_bench.logging import get_logger
from flashinfer_bench.utils import dtype_str_to_python_dtype, dtype_str_to_torch_dtype

from .builtin.configs import FULL_TRACING_CONFIGS
from .config import TracingConfig
from .filter_policy import FilterPolicy
from .workload_entry import WorkloadEntry

logger = get_logger("TracingRuntime")


class TracingRuntime:
    """Process-wide singleton tracer for workload collection."""

    def __init__(
        self,
        trace_set: TraceSet,
        tracing_configs: Optional[Dict[str, TracingConfig]] = None,
        prev_tracing_runtime: Optional["TracingRuntime"] = None,
    ):
        """
        Initialize the tracing runtime.

        Parameters
        ----------
        dataset_path : Optional[str]
            Path to the dataset/traceset directory. Default is the environment variable
            FIB_DATASET_PATH if it exists, or `~/.cache/flashinfer_bench/dataset`.
        tracing_configs : Optional[Dict[str, TracingConfig]]
            A set of tracing configs. Default is `fib.tracing.builtin_config.fib_full_tracing`.
        prev_tracing_runtime : Optional[TracingRuntime]
            The previous tracing runtime. Will be used in the __exit__ method.
        """
        self._trace_set = trace_set

        tracing_configs_non_null = tracing_configs if tracing_configs else FULL_TRACING_CONFIGS
        self._tracing_configs = tracing_configs_non_null

        self._prev_runtime = prev_tracing_runtime

        # Global order counter
        self.order_counter = 0

        # Thread safety
        self._lock = threading.Lock()

        # CUDA Graph support
        self._cuda_graph_entries: List[WorkloadEntry] = []
        self._in_cuda_graph = False

        # Create independent filter policy instances for each definition
        # This ensures state isolation between definitions and runtime instances
        self._filter_policies: Dict[str, FilterPolicy] = {}
        for def_name, config in self._tracing_configs.items():
            self._filter_policies[def_name] = config.create_filter_policy()

        # Validate config keys exist in definitions
        for def_name in self._tracing_configs:
            if def_name not in self._trace_set.definitions:
                logger.warning(f"Tracing config found for unknown definition: {def_name}")

        # Register cleanup handlers
        atexit.register(self._exit_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        logger.debug("TracingRuntime Initialized")
        logger.debug(f"  TraceSet root path: {self._trace_set.root}")
        logger.debug(f"  Tracing configs: {len(self._tracing_configs)} definitions configured")

        from flashinfer_bench.integration.flashinfer import install_flashinfer_integrations

        install_flashinfer_integrations()

    def collect(self, def_name: str, args: Iterable[Any]):
        """
        Record a workload for later serialization to disk.

        Parameters
        ----------
        def_name : str
            Name of the workload definition to trace.
        args : Iterable[Any]
            Positional arguments following the definition's input order.

        Notes
        -----
        Entries are buffered in memory until flush() is called.
        """
        logger.debug(f"Tracing '{def_name}'")
        tracing_config = self._tracing_configs.get(def_name)
        if tracing_config is None:
            logger.error(f"Tracing config not configured for {def_name}, skipping")
            return

        if def_name not in self._trace_set.definitions:
            logger.error(f"Definition {def_name} not found, skipping")
            return

        definition = self._trace_set.definitions[def_name]
        input_names = list(definition.inputs.keys())

        if len(args) != len(definition.inputs):
            logger.error(f"Expected {len(input_names)} args for {def_name}, got {len(args_list)}")
            return

        # Build runtime_args dict and extract input shapes
        runtime_args = dict(zip(input_names, args_list))
        input_shapes = [arg.shape if isinstance(arg, torch.Tensor) else None for arg in args_list]

        # Infer variable axes using Definition.get_var_values
        try:
            axes = definition.get_var_values(input_shapes)
        except ValueError as e:
            logger.error(f"Error inferring axes for {def_name}: {e}")
            return

        # Determine which inputs to dump
        input_names_to_dump = tracing_config.get_inputs_to_dump(runtime_args)
        inputs_to_dump: Dict[str, torch.Tensor] = {}

        for name in input_names_to_dump:
            converted = self._convert_arg_to_tensor(definition, name, runtime_args[name])
            if converted is None:
                return
            inputs_to_dump[name] = converted

        entry = WorkloadEntry(
            def_name=def_name, axes=axes, inputs_to_dump=inputs_to_dump, order=self.order_counter
        )

        with self._lock:
            if self._in_cuda_graph:
                # Deferred snapshot for CUDA Graph replay
                self._cuda_graph_entries.append(entry)
            else:
                # Submit entry directly to filter policy for online filtering
                filter_policy = self._filter_policies.get(def_name)
                if filter_policy is not None:
                    filter_policy.submit(entry)

            self.order_counter += 1

    def _convert_arg_to_tensor(
        self, definition: Definition, name: str, val: Any
    ) -> Optional[torch.Tensor]:
        """Convert a runtime argument to a tensor for dumping.

        Parameters
        ----------
        definition : Definition
            The workload definition containing input specifications.
        name : str
            Name of the argument to convert.
        val : Any
            The runtime argument to convert.

        Returns
        -------
        Optional[torch.Tensor]
            The converted tensor. None if conversion fails.

        Notes
        -----
        Shape validation is already done in Definition.get_var_values().
        """
        spec = definition.inputs[name]
        torch_dtype = dtype_str_to_torch_dtype(spec.dtype)

        # Scalar input
        if spec.shape is None:
            python_dtype = dtype_str_to_python_dtype(spec.dtype)
            if not isinstance(val, python_dtype):
                logger.error(
                    f'Input "{name}" must be Python scalar of type {python_dtype.__name__},'
                    f" but got {type(val).__name__}"
                )
                return None
            val = torch.tensor(val, dtype=torch_dtype)

        # Tensor input
        else:
            if not isinstance(val, torch.Tensor):
                logger.error(f'Input "{name}" must be a tensor (got {type(val).__name__})')
                return None
            if val.dtype != torch_dtype:
                logger.error(f'Input "{name}" must have dtype {torch_dtype}, got {val.dtype}')
                return None

        if not self._in_cuda_graph:
            val = val.detach().cpu().clone()

        return val

    def _snapshot_graph_tensors(self):
        """Snapshot tensors from CUDA Graph entries to CPU memory.

        Notes
        -----
        This method is called automatically when exiting cuda_graph_scope().
        It synchronizes CUDA execution, creates CPU copies of all deferred
        tensors from CUDA Graph entries, and submits them to filter policies.
        The deferred entries buffer is cleared after processing.
        """
        # Synchronize CUDA before taking snapshots
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        for entry in self._cuda_graph_entries:
            # Create CPU snapshots
            snapshot = {}
            for name, tensor in entry.inputs_to_dump.items():
                if isinstance(tensor, torch.Tensor):
                    snapshot[name] = tensor.detach().cpu().clone()
                else:
                    snapshot[name] = tensor
            entry.cuda_graph_snapshot = snapshot
            entry.inputs_to_dump = snapshot

            # Submit to filter policy
            filter_policy = self._filter_policies.get(entry.def_name)
            if filter_policy is not None:
                filter_policy.submit(entry)

        self._cuda_graph_entries.clear()

    def _convert_workload_entry_to_trace(self, entry: WorkloadEntry) -> Optional[Trace]:
        """Convert a workload entry to a trace.

        Parameters
        ----------
        entry : WorkloadEntry
            The workload entry to convert.

        Returns
        -------
        Optional[Trace]
            The converted trace. None if conversion fails, such as system error occuring during
            saving tensors.
        """
        workload_uuid = str(uuid.uuid4())
        inputs: Dict[str, InputSpec] = {}

        # Dump picked input tensors
        if len(entry.inputs_to_dump) > 0:
            try:
                save_path = self._trace_set.add_workload_blob_tensor(
                    entry.def_name, workload_uuid, entry.inputs_to_dump
                )
            except Exception as e:
                logger.error(f"Failed to save tensors for {entry.def_name}: {e}")
                return None
            for name in entry.inputs_to_dump:
                inputs[name] = SafetensorsInput(path=save_path, tensor_key=name)

        # Set random for non-picked input tensors
        definition = self._trace_set.definitions[entry.def_name]
        for name in definition.inputs.keys():
            if name not in inputs:
                inputs[name] = RandomInput()

        # Return the trace
        return Trace(
            definition=entry.def_name,
            workload=Workload(axes=entry.axes, inputs=inputs, uuid=workload_uuid),
            solution=None,
            evaluation=None,
        )

    def flush(self):
        """
        Drain selected entries from filter policies and write to disk.
        """
        # Stats
        num_selected_entries = 0
        num_dump_errors = 0

        if self._in_cuda_graph:
            raise RuntimeError("Cannot flush during CUDA Graph replay")

        # Drain entries from each filter policy and convert to traces
        for filter_policy in self._filter_policies.values():
            # Drain selected entries from policy
            selected_entries = filter_policy.drain()

            if len(selected_entries) == 0:
                continue

            traces_to_dump: List[Trace] = []
            for entry in selected_entries:
                trace = self._convert_workload_entry_to_trace(entry)
                if trace is None:
                    num_dump_errors += 1
                else:
                    traces_to_dump.append(trace)

            self._trace_set.add_workload_traces(traces_to_dump)
            num_selected_entries += len(selected_entries)

        # Log stats
        logger.info(
            f"Flush done. {num_selected_entries} entries selected, {num_dump_errors} dump errors"
        )

    def _exit_handler(self):
        """Exit handler. Stores all buffered trace entries to disk."""
        try:
            self.flush()
        except Exception as e:
            logger.error(f"Flush failed at exit: {e}")

    def _signal_handler(self, signum, frame):
        """Signal handler for SIGTERM/SIGINT. Stores all buffered trace entries to disk.

        Parameters
        ----------
        signum : int
            Signal number that triggered the handler.
        frame : FrameType
            Current stack frame when the signal was received.
        """
        try:
            self.flush()
        except Exception as e:
            logger.error(f"Flush failed at signal handler: {e}")

    def __enter__(self) -> "TracingRuntime":
        """Context manager entry point.

        Returns
        -------
        TracingRuntime
            Returns self to enable context manager usage.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit point. Flushes buffered traces and restores previous runtime.

        Parameters
        ----------
        exc_type : Optional[Type[BaseException]]
            Exception type if an exception occurred, None otherwise.
        exc_val : Optional[BaseException]
            Exception instance if an exception occurred, None otherwise.
        exc_tb : Optional[TracebackType]
            Traceback object if an exception occurred, None otherwise.

        Returns
        -------
        bool
            Always returns False to propagate any exceptions that occurred.
        """
        self.flush()
        set_tracing_runtime(self._prev_runtime)
        return False


def _init_tracing_runtime_from_env() -> Optional["TracingRuntime"]:
    """Initialize the global tracing runtime instance from environment variables."""
    fib_enable_tracing = get_fib_enable_tracing()
    if not fib_enable_tracing:
        return None
    fib_dataset_path = get_fib_dataset_path()
    trace_set = TraceSet.from_path(fib_dataset_path)
    tracing_configs = FULL_TRACING_CONFIGS
    return TracingRuntime(trace_set, tracing_configs, None)


# Global singleton tracing runtime instance
_global_tracing_runtime: Optional["TracingRuntime"] = _init_tracing_runtime_from_env()


def get_tracing_runtime() -> Optional["TracingRuntime"]:
    """Get the global tracing runtime instance.

    Returns
    -------
    TracingRuntime
        The global runtime instance.
    """
    return _global_tracing_runtime


def set_tracing_runtime(rt: Optional["TracingRuntime"]) -> None:
    """Set the global tracing runtime instance.

    Parameters
    ----------
    rt : Optional[TracingRuntime]
        The runtime instance to set, or None to clear the global runtime.
    """
    global _global_tracing_runtime
    _global_tracing_runtime = rt


# TODO: Fix cuda graph support
class CudaGraphTracingRuntime:
    """Context manager for tracing operations within CUDA graphs.

    This class provides a context manager that temporarily enables CUDA graph mode
    in the associated TracingRuntime. When entering the context, it sets the runtime
    to CUDA graph mode, and when exiting, it disables the mode and snapshots any
    graph tensors for tracing purposes.
    """

    def __init__(self, tracing_runtime: "TracingRuntime"):
        """Initialize the CUDA graph tracing runtime context manager.

        Parameters
        ----------
        tracing_runtime : TracingRuntime
            The TracingRuntime instance to manage CUDA graph state for.
        """
        self.tracing_runtime = tracing_runtime

    def __enter__(self) -> "CudaGraphTracingRuntime":
        """Enter the CUDA graph tracing context.

        Sets the associated TracingRuntime to CUDA graph mode under lock protection.

        Returns
        -------
        CudaGraphTracingRuntime
            Returns self to enable context manager usage.
        """
        with self.tracing_runtime._lock:
            self.tracing_runtime._in_cuda_graph = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit the CUDA graph tracing context.

        Disables CUDA graph mode and snapshots graph tensors for tracing.

        Parameters
        ----------
        exc_type : Optional[Type[BaseException]]
            Exception type if an exception occurred, None otherwise.
        exc_val : Optional[BaseException]
            Exception instance if an exception occurred, None otherwise.
        exc_tb : Optional[TracebackType]
            Traceback object if an exception occurred, None otherwise.
        """
        with self.tracing_runtime._lock:
            self.tracing_runtime._in_cuda_graph = False
            self.tracing_runtime._snapshot_graph_tensors()
        return False
