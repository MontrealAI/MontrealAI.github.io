#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║     🪐🔮  AGENTIC ALPHA EXPLORER (INFINITE) - ASI SUPERINTELLIGENCE ERA  🔮🪐    ║
╚═══════════════════════════════════════════════════════════════════════════════╝

EXPERIENCE NEXT-LEVEL COGNITIVE DISCOVERY:

This single-file script performs an infinite BFS-style exploration of concepts, 
optionally coordinated by an “agentic” parallel approach (inspired by agentic_treesearch).
By default, it calls out to a local Ollama instance (http://localhost:11434) to generate 
sub-concepts from a root concept. The expansions continue indefinitely until you press Ctrl+C.

ASCEND INTO THE ASI-ERA:
    - **Immersive Terminal FX** via Rich: dynamic color palette, cosmic ASCII trees,
      animated spinners, and more to evoke superintelligent aesthetics.
    - **Infinite BFS** or **Agentic** expansions: 
      Classic BFS or a minimal parallel agent approach controlling expansions.
    - **Production-Ready**:
      - Single-file code, robust error handling, session auto-saving, graceful shutdown.
      - Straightforward CLI with intuitive flags.
      - Cross-platform (Windows/macOS/Linux) support with zero manual tweaks needed.
    - **Highly Customizable**:
      - Switch LLM backends or adapt to local vs. cloud models easily (default: Ollama).
      - Diversify expansions by adjusting `--diversity`.

DEPENDENCIES:
    - Python 3.7+ 
    - `rich` (for advanced terminal UI)
    - `requests`, `networkx`, `dataclasses_json`, `numpy`, `pandas`, `PyYAML`, `humanize`
    - A local or remote LLM backend (default: Ollama). 
      Or adapt the code to OpenAI, HuggingFace, etc.

AUTHOR:
    A specialized, well-funded, and futuristic team from the pinnacle 
    of AI engineering & design, ensuring a unique and powerful user experience.

USAGE:
    python agentic_alpha_explorer.py --root="Quantum Consciousness" --diversity=1.0 --agentic

    (Press Ctrl+C at any time to stop expansions.)

"""

import sys

# ------------------------------------------------------------------------------
# Safely import Rich (or bail out gracefully if missing).
# ------------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.theme import Theme
    from rich.table import Table
    from rich.tree import Tree
    from rich import box
    from rich.panel import Panel
    from rich.spinner import Spinner
    from rich.markdown import Markdown
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Prompt
    from rich.style import Style as RichStyle
    from rich.traceback import install as rich_traceback_install
except ImportError:
    print(
        "\nERROR: The 'rich' library is not installed.\n"
        "Please install it by running:\n"
        "    pip install rich\n"
        "Then re-run this script.\n"
    )
    sys.exit(1)

import argparse
import json
import os
import random
import shutil
import textwrap
import time
import logging
import traceback
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from dataclasses_json import DataClassJsonMixin
from enum import Enum, auto
from functools import total_ordering
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Callable, Set

import requests
import humanize
import yaml
import numpy as np
import pandas as pd
import networkx as nx

# ------------------------------------------------------------------------------
# Configure a custom Rich theme + advanced traceback
# ------------------------------------------------------------------------------
custom_theme = Theme({
    "info": "dim cyan",
    "warning": "bold yellow",
    "error": "bold red",
    "success": "bold green",
    "title": "bold magenta"
})
console = Console(theme=custom_theme)
rich_traceback_install(show_locals=False)

# ------------------------------------------------------------------------------
# Global Logging
# ------------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ------------------------------------------------------------------------------
# Default Configurations
# ------------------------------------------------------------------------------
DEFAULT_MODEL = "llama3"
DEFAULT_ROOT_CONCEPT = "Consciousness Cascades"
DEFAULT_DIVERSITY_BIAS = 1.0
DEFAULT_MAX_DEPTH = 4   # Not actually used in infinite BFS, retained for legacy
SLEEP_DURATION = 0.8

# ------------------------------------------------------------------------------
# Clear Terminal
# ------------------------------------------------------------------------------
def clear_terminal():
    """
    Clears the console in a cross-platform manner,
    with Rich for a smooth user experience.
    """
    console.clear()

# ------------------------------------------------------------------------------
# Agentic BFS-like Structures
# ------------------------------------------------------------------------------
@dataclass
class FunctionSpec:
    """
    Stub for advanced function calls or schemas used in agentic expansions.
    """
    name: str
    json_schema: dict
    description: str

    def __post_init__(self):
        import jsonschema
        jsonschema.Draft7Validator.check_schema(self.json_schema)


@total_ordering
@dataclass
class MetricValue:
    """
    A numeric or dict-based metric for BFS or code evaluation. 
    Compareable: we can see which expansions are "best."
    """
    value: float | int | dict | None
    maximize: bool | None = False
    name: str | None = None
    description: str | None = None

    def __post_init__(self):
        if self.value is not None:
            if isinstance(self.value, dict):
                if "metric_names" in self.value:
                    for m in self.value["metric_names"]:
                        for d in m["data"]:
                            if d["final_value"] is not None:
                                d["final_value"] = float(d["final_value"])
                            if d["best_value"] is not None:
                                d["best_value"] = float(d["best_value"])
            else:
                self.value = float(self.value)

    def _should_maximize(self) -> bool:
        return bool(self.maximize)

    def get_mean_value(self) -> float:
        """
        A single float value for comparisons. Averages sub-metrics if dict-based.
        """
        if self.value is None:
            return float("nan")
        if isinstance(self.value, dict):
            if "metric_names" in self.value:
                aggregated = []
                for m in self.value["metric_names"]:
                    subvals = [d["final_value"] for d in m["data"] if d["final_value"] is not None]
                    if subvals:
                        aggregated.extend(subvals)
                return float(np.mean(aggregated)) if aggregated else float("nan")
            vals = [v for v in self.value.values() if v is not None]
            return float(np.mean(vals)) if vals else float("nan")
        return float(self.value)

    def __gt__(self, other):
        if self.value is None:
            return False
        if other.value is None:
            return True
        s_val = self.get_mean_value()
        o_val = other.get_mean_value()
        if s_val == o_val:
            return False
        return s_val > o_val if self._should_maximize() else s_val < o_val

    def __eq__(self, other):
        if not isinstance(other, MetricValue):
            return False
        if self.value is None and other.value is None:
            return True
        if self.value is None or other.value is None:
            return False
        return self.value == other.value

    def __str__(self):
        """
        Nicely formatted string representation of the metric.
        """
        if isinstance(self.value, dict):
            if "metric_names" in self.value:
                parts = []
                for m in self.value["metric_names"]:
                    arrow = "↓" if m["lower_is_better"] else "↑"
                    vs = []
                    for d in m["data"]:
                        vs.append(
                            f'{d["dataset_name"]}:(final={d["final_value"]:.4f},best={d["best_value"]:.4f})'
                        )
                    parts.append(f'{m["metric_name"]}{arrow}[{",".join(vs)}]')
                return "Metrics(" + "; ".join(parts) + ")"
            arrow = "↑" if self.maximize else "↓"
            val_str = ",".join(f"{k}:{v:.4f}" for k, v in self.value.items() if v is not None)
            mean_val = np.mean([v for v in self.value.values() if v is not None])
            return f"Metric{arrow}({self.name})[{val_str}](mean={mean_val:.4f})"

        arrow = "?" if self.maximize is None else ("↑" if self.maximize else "↓")
        nm = f"({self.name})" if self.name else ""
        return f"Metric{arrow}{nm}({self.get_mean_value():.4f})"


@dataclass
class WorstMetricValue(MetricValue):
    """
    Fallback if code crashed or no metric is available.
    """
    value = None


@dataclass
class ExecutionResult:
    """
    Captures output from child process: stdout, time, and exceptions if any.
    """
    term_out: List[str]
    exec_time: float
    exc_type: str | None
    exc_info: dict | None = None
    exc_stack: List[Tuple] | None = None


def exception_summary(e, working_dir, exec_file_name):
    tb_lines = traceback.format_exception(e)
    tb_str = "".join(line for line in tb_lines if "treesearch/" not in line and "importlib" not in line)
    tb_str = tb_str.replace(str(Path(working_dir) / exec_file_name), exec_file_name)
    exc_info = {}
    if hasattr(e, "args"):
        exc_info["args"] = [str(a) for a in e.args]
    for att in ["name", "msg", "obj"]:
        if hasattr(e, att):
            exc_info[att] = str(getattr(e, att))
    tb = traceback.extract_tb(e.__traceback__)
    exc_stack = [(t.filename, t.lineno, t.name, t.line) for t in tb]
    return tb_str, e.__class__.__name__, exc_info, exc_stack


class RedirectQueue:
    """
    Utility that redirects child process stdout/stderr to a queue.
    """
    def __init__(self, q):
        self.queue = q

    def write(self, msg):
        self.queue.put(msg)

    def flush(self):
        pass


class Interpreter:
    """
    Minimal code runner in a child process with timeouts + logging.
    """
    def __init__(
        self,
        working_dir: Path | str,
        timeout=3600,
        format_tb_ipython=False,
        agent_file_name="runfile.py",
        env_vars: dict[str, str] = {},
    ):
        self.working_dir = Path(working_dir).resolve()
        assert self.working_dir.exists()
        self.timeout = timeout
        self.format_tb_ipython = format_tb_ipython
        self.agent_file_name = agent_file_name
        self.env_vars = env_vars
        self.process = None

    def child_proc_setup(self, result_outq):
        import shutup
        shutup.mute_warnings()
        for k, v in self.env_vars.items():
            os.environ[k] = v
        os.chdir(str(self.working_dir))
        sys.path.append(str(self.working_dir))
        sys.stdout = sys.stderr = RedirectQueue(result_outq)

    def _run_session(self, code_inq, result_outq, event_outq):
        """
        Core loop in child process: read code from code_inq, run it, return results.
        """
        self.child_proc_setup(result_outq)
        global_scope = {}
        while True:
            code = code_inq.get()
            os.chdir(str(self.working_dir))
            with open(self.agent_file_name, "w") as f:
                f.write(code)
            event_outq.put(("state:ready",))

            try:
                exec(compile(code, self.agent_file_name, "exec"), global_scope)
            except BaseException as e:
                tb_str, e_cls_name, exc_info, exc_stack = exception_summary(
                    e, self.working_dir, self.agent_file_name
                )
                result_outq.put(tb_str)
                if e_cls_name == "KeyboardInterrupt":
                    e_cls_name = "TimeoutError"
                event_outq.put(("state:finished", e_cls_name, exc_info, exc_stack))
            else:
                event_outq.put(("state:finished", None, None, None))
            result_outq.put("<|EOF|>")

    def create_process(self):
        from multiprocessing import Process, Queue
        self.code_inq, self.result_outq, self.event_outq = Queue(), Queue(), Queue()
        self.process = Process(
            target=self._run_session,
            args=(self.code_inq, self.result_outq, self.event_outq),
        )
        self.process.start()

    def _drain_queues(self):
        """
        Purge leftover items from queues, used before process termination.
        """
        while not self.result_outq.empty():
            try:
                self.result_outq.get_nowait()
            except:
                break
        while not self.event_outq.empty():
            try:
                self.event_outq.get_nowait()
            except:
                break
        while not self.code_inq.empty():
            try:
                self.code_inq.get_nowait()
            except:
                break

    def cleanup_session(self):
        """
        Terminate the child process and clean up queues.
        """
        if not self.process:
            return
        self.process.terminate()
        self._drain_queues()
        self.process.join(timeout=2)
        if self.process.exitcode is None:
            logger.warning("Child process did not exit, forcing kill.")
            self.process.kill()
            self._drain_queues()
            self.process.join(timeout=2)
        self.process.close()
        self.process = None

    def run(self, code: str, reset_session=True) -> ExecutionResult:
        import time
        import queue as pyqueue

        if reset_session:
            if self.process:
                self.cleanup_session()
            self.create_process()
        else:
            assert self.process and self.process.is_alive()

        self.code_inq.put(code)
        child_in_overtime = False
        start_time = time.time()

        # Wait for "state:ready"
        while True:
            try:
                st = self.event_outq.get(timeout=10)
                assert st[0] == "state:ready", st
                break
            except pyqueue.Empty:
                if not self.process.is_alive():
                    msg = "Child process died unexpectedly before 'ready'."
                    while not self.result_outq.empty():
                        logger.error(f"[Child Output] {self.result_outq.get()}")
                    raise RuntimeError(msg)

        # Wait for "state:finished"
        while True:
            try:
                st = self.event_outq.get(timeout=1)
                assert st[0] == "state:finished", st
                exec_time = time.time() - start_time
                e_cls_name, exc_info, exc_stack = st[1], st[2], st[3]
                break
            except pyqueue.Empty:
                if not child_in_overtime and not self.process.is_alive():
                    msg = "Child process died unexpectedly during execution."
                    while not self.result_outq.empty():
                        logger.error(f"[Child Output] {self.result_outq.get()}")
                    raise RuntimeError(msg)
                if self.timeout:
                    elapsed = time.time() - start_time
                    if elapsed > self.timeout:
                        os.kill(self.process.pid, 2)  # SIGINT
                        child_in_overtime = True
                        if elapsed > self.timeout + 60:
                            self.cleanup_session()
                            e_cls_name = "TimeoutError"
                            exec_time = self.timeout
                            break

        # Gather output from child
        output = []
        while not self.result_outq.empty() or not output or output[-1] != "<|EOF|>":
            output.append(self.result_outq.get())
        output.pop()  # remove <|EOF|>

        if e_cls_name == "TimeoutError":
            output.append(f"TimeoutError: exceeded {humanize.naturaldelta(self.timeout)}")
        else:
            output.append(f"Execution time: {humanize.naturaldelta(exec_time)}")

        return ExecutionResult(output, exec_time, e_cls_name, exc_info, exc_stack)


class MinimalAgent:
    """
    Stub BFS or debug agent. Real usage might connect to an LLM to generate expansions or code.
    """
    def __init__(
        self,
        task_desc,
        cfg,
        memory_summary=None,
        evaluation_metrics=None,
        stage=None,
        stage_name=None,
    ):
        self.task_desc = task_desc
        self.memory_summary = memory_summary
        self.cfg = cfg
        self.evaluation_metrics = evaluation_metrics
        self.stage_name = stage_name

    def plan_and_code_query(self, prompt, retries=3) -> Tuple[str, str]:
        import re
        c = None
        for _ in range(retries):
            c = "Plan in natural language\n```python\nprint('Hello BFS from MinimalAgent!')\n```"
            match = re.search(r"```python\s*(.*?)\s*```", c, re.DOTALL)
            if match:
                code_block = match.group(1)
                plain_text_plan = c[: match.start()].strip()
                return plain_text_plan, code_block
        return "", ""

    def parse_exec_result(self, node, exec_result, workspace):
        node.absorb_exec_result(exec_result)
        node.analysis = "Bug detected" if node.exc_type else "OK"
        node.is_buggy = bool(node.exc_type)


class GPUManager:
    """
    Manages GPU usage across parallel processes if relevant. 
    Here it's mostly a stub returning 1 GPU for demonstration.
    """
    def __init__(self, num_gpus: int):
        self.num_gpus = num_gpus
        self.available_gpus = set(range(num_gpus))
        self.gpu_assignments = {}

    def acquire_gpu(self, process_id: str) -> int:
        if not self.available_gpus:
            raise RuntimeError("No GPUs available.")
        g = min(self.available_gpus)
        self.available_gpus.remove(g)
        self.gpu_assignments[process_id] = g
        return g

    def release_gpu(self, process_id: str):
        if process_id in self.gpu_assignments:
            g = self.gpu_assignments[process_id]
            self.available_gpus.add(g)
            del self.gpu_assignments[process_id]


def get_gpu_count() -> int:
    """
    Returns 1 by default for demonstration.
    """
    return 1


@dataclass(eq=False)
class Node(DataClassJsonMixin):
    """
    BFS Node for expansions. Typically, `plan` holds the concept text.
    """
    plan: str = ""
    overall_plan: str = ""
    code: str = ""
    plot_code: str = None
    plot_plan: str = None
    step: int = None
    id: str = field(default_factory=lambda: hex(random.getrandbits(64))[2:])
    ctime: float = field(default_factory=lambda: time.time())
    parent: Optional["Node"] = None
    children: set = field(default_factory=set)
    exp_results_dir: str = None

    _term_out: List[str] = None
    exec_time: float = None
    exc_type: str | None = None
    exc_info: dict | None = None
    exc_stack: List[Tuple] = None

    parse_metrics_plan: str = ""
    parse_metrics_code: str = ""
    parse_term_out: List[str] = None
    parse_exc_type: str | None = None
    parse_exc_info: dict | None = None
    parse_exc_stack: List[Tuple] = None

    plot_term_out: List[str] = None
    plot_exec_time: float = None
    plot_exc_type: str | None = None
    plot_exc_info: dict | None = None
    plot_exc_stack: List[Tuple] = None

    analysis: str = None
    metric: MetricValue = None
    is_buggy: bool = None
    is_buggy_plots: bool = None
    plot_data: dict = field(default_factory=dict)
    plots_generated: bool = False
    plots: List[str] = field(default_factory=list)
    plot_paths: List[str] = field(default_factory=list)
    plot_analyses: List[Any] = field(default_factory=list)
    vlm_feedback_summary: List[str] = field(default_factory=list)
    datasets_successfully_tested: List[str] = field(default_factory=list)
    exec_time_feedback: str = ""
    ablation_name: str = None
    hyperparam_name: str = None
    is_seed_node: bool = False
    is_seed_agg_node: bool = False

    def __post_init__(self):
        if isinstance(self.children, list):
            self.children = set(self.children)
        if self.parent and isinstance(self.parent, Node):
            self.parent.children.add(self)

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    @property
    def stage_name(self):
        if self.parent is None:
            return "draft"
        return "debug" if self.parent.is_buggy else "improve"

    def absorb_exec_result(self, exec_result: ExecutionResult):
        self._term_out = exec_result.term_out
        self.exec_time = exec_result.exec_time
        self.exc_type = exec_result.exc_type
        self.exc_info = exec_result.exc_info
        self.exc_stack = exec_result.exc_stack

    def absorb_plot_exec_result(self, plot_exec_result: ExecutionResult):
        self.plot_term_out = plot_exec_result.term_out
        self.plot_exec_time = plot_exec_result.exec_time
        self.plot_exc_type = plot_exec_result.exc_type
        self.plot_exc_info = plot_exec_result.exc_info
        self.plot_exc_stack = plot_exec_result.exc_stack

    @property
    def term_out(self) -> str:
        """
        Gather captured stdout/stderr, truncated if large.
        """
        def trim_output(s, limit=5100, keep=2500):
            if len(s) > limit:
                return s[:keep] + "\n... [TRUNCATED] ...\n" + s[-keep:]
            return s
        text = "".join(self._term_out) if self._term_out else ""
        return trim_output(text)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def debug_depth(self) -> int:
        if self.stage_name != "debug":
            return 0
        return self.parent.debug_depth + 1

    def to_dict(self) -> Dict:
        """
        Convert this Node to a dictionary for JSON serialization.
        """
        cpth = None
        if self.exp_results_dir:
            cpth = str(Path(self.exp_results_dir).resolve().relative_to(os.getcwd()))

        pm = []
        if self.plot_analyses:
            for item in self.plot_analyses:
                copy_item = item.copy()
                if copy_item.get("plot_path"):
                    copy_item["plot_path"] = str(Path(copy_item["plot_path"]).resolve().relative_to(os.getcwd()))
                pm.append(copy_item)

        return {
            "code": self.code,
            "plan": self.plan,
            "overall_plan": self.overall_plan,
            "plot_code": self.plot_code,
            "plot_plan": self.plot_plan,
            "step": self.step,
            "id": self.id,
            "ctime": self.ctime,
            "_term_out": self._term_out,
            "parse_metrics_plan": self.parse_metrics_plan,
            "parse_metrics_code": self.parse_metrics_code,
            "parse_term_out": self.parse_term_out,
            "parse_exc_type": self.parse_exc_type,
            "parse_exc_info": self.parse_exc_info,
            "parse_exc_stack": self.parse_exc_stack,
            "exec_time": self.exec_time,
            "exc_type": self.exc_type,
            "exc_info": self.exc_info,
            "exc_stack": self.exc_stack,
            "analysis": self.analysis,
            "exp_results_dir": cpth,
            "metric": {
                "value": self.metric.value if self.metric else None,
                "maximize": self.metric.maximize if self.metric else None,
                "name": self.metric.name if self.metric else None,
                "description": self.metric.description if self.metric else None,
            },
            "is_buggy": self.is_buggy,
            "is_buggy_plots": self.is_buggy_plots,
            "parent_id": None if not self.parent else self.parent.id,
            "children": [c.id for c in self.children],
            "plot_data": self.plot_data,
            "plots_generated": self.plots_generated,
            "plots": self.plots,
            "plot_paths": [
                str(Path(p).resolve().relative_to(os.getcwd())) for p in self.plot_paths
            ],
            "plot_analyses": pm,
            "vlm_feedback_summary": self.vlm_feedback_summary,
            "datasets_successfully_tested": self.datasets_successfully_tested,
            "ablation_name": self.ablation_name,
            "hyperparam_name": self.hyperparam_name,
            "is_seed_node": self.is_seed_node,
            "is_seed_agg_node": self.is_seed_agg_node,
            "exec_time_feedback": self.exec_time_feedback,
        }

    @classmethod
    def from_dict(cls, data: Dict, journal=None) -> "Node":
        pid = data.pop("parent_id", None)
        _children = data.pop("children", [])
        m = data.pop("metric", None)
        if m:
            if isinstance(m, dict):
                data["metric"] = MetricValue(
                    value=m["value"],
                    maximize=m["maximize"],
                    name=m["name"],
                    description=m["description"],
                )
            else:
                data["metric"] = WorstMetricValue() if data.get("is_buggy") else MetricValue(m)

        node = cls(**data)
        if journal and pid:
            parent_node = journal.get_node_by_id(pid)
            if parent_node:
                node.parent = parent_node
                parent_node.children.add(node)
        return node


@dataclass
class Journal:
    """
    Stores BFS expansions for introspection or reloading sessions.
    """
    nodes: List[Node] = field(default_factory=list)

    def __getitem__(self, idx: int) -> Node:
        return self.nodes[idx]

    def __len__(self) -> int:
        return len(self.nodes)

    def append(self, node: Node) -> None:
        node.step = len(self.nodes)
        self.nodes.append(node)

    @property
    def draft_nodes(self) -> List[Node]:
        return [n for n in self.nodes if not n.parent]

    @property
    def buggy_nodes(self) -> List[Node]:
        return [n for n in self.nodes if n.is_buggy]

    @property
    def good_nodes(self) -> List[Node]:
        return [n for n in self.nodes if not n.is_buggy and not n.is_buggy_plots]

    def get_node_by_id(self, node_id: str) -> Optional[Node]:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def get_best_node(self, only_good=True):
        cands = self.good_nodes if only_good else self.nodes
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]
        best = cands[0]
        for n in cands:
            if n.metric and n.metric > best.metric:
                best = n
        return best

    def generate_summary(self, include_code=False) -> str:
        if not self.nodes:
            return "No expansions performed."
        bn = self.get_best_node()
        return f"Summary: total={len(self.nodes)} expansions; best_node=({bn.id if bn else 'None'})"


class ParallelAgent:
    """
    A minimal BFS agent that can handle expansions in parallel. 
    """
    def __init__(
        self,
        task_desc: str,
        cfg,
        journal: Journal,
        stage_name=None,
        best_stage3_node=None,
        best_stage2_node=None,
        best_stage1_node=None,
    ):
        self.task_desc = task_desc
        self.cfg = cfg
        self.journal = journal
        self.stage_name = stage_name
        self.best_stage3_node = best_stage3_node
        self.best_stage2_node = best_stage2_node
        self.best_stage1_node = best_stage1_node

        self.num_workers = cfg["agent"]["num_workers"]
        self.num_gpus = get_gpu_count()
        self.gpu_manager = GPUManager(self.num_gpus) if self.num_gpus > 0 else None
        self.timeout = cfg["exec"]["timeout"]
        self.executor = ProcessPoolExecutor(max_workers=self.num_workers)
        self._is_shutdown = False

    def cleanup(self):
        if not self._is_shutdown:
            try:
                if self.gpu_manager:
                    for pid in list(self.gpu_manager.gpu_assignments.keys()):
                        self.gpu_manager.release_gpu(pid)
                self.executor.shutdown(wait=False, cancel_futures=True)
            except:
                pass
            self._is_shutdown = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def step(self, exec_callback):
        pass

# ------------------------------------------------------------------------------
# CONCEPT EXPLORER CORE
# ------------------------------------------------------------------------------
class ConceptExplorer:
    """
    A BFS-based concept explorer with optional agentic expansions. 
    By default, expansions come from a local Ollama server.
    Press Ctrl+C to stop at any time.
    """
    def __init__(self, model=DEFAULT_MODEL):
        self.graph = nx.DiGraph()
        self.seen_concepts = set()
        self.last_added = None
        self.current_concept = None
        self.model = model

        self.term_width, self.term_height = shutil.get_terminal_size((80, 24))
        self.last_tree_update_time = 0
        self.MIN_UPDATE_INTERVAL = 0.5

    def get_available_models(self) -> List[str]:
        """
        Attempt to query local Ollama for models.
        If not using Ollama, adapt to your own LLM or skip.
        """
        url = "http://localhost:11434/api/tags"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json().get("models", [])
            return [m["name"] for m in data]
        except Exception as e:
            console.print(f"[error]Error contacting Ollama: {e}[/error]")
            return []

    def check_model_availability(self) -> bool:
        available = self.get_available_models()
        if not available:
            return False
        if self.model in available:
            return True
        for model_name in available:
            if model_name.startswith(f"{self.model}:"):
                self.model = model_name
                return True
        return False

    def strip_thinking_tags(self, response: str) -> str:
        """
        Remove <think>...</think> from streaming output if present.
        """
        return response.replace("<think>", "").replace("</think>", "")

    def _update_thinking_block(self, text: str, state: Dict[str, Any]):
        """
        Animate partial LLM "thinking" text with an ephemeral display.
        """
        lines = textwrap.wrap(text.strip(), width=self.term_width)
        if not state["printed_brain_emoji"]:
            lines.insert(0, "🧠")
            state["printed_brain_emoji"] = True

        if len(lines) > 6:
            lines = lines[:6]
            lines[-1] += "..."

        if lines == state["last_printed_block"]:
            return

        # Clear old lines
        for _ in range(state["printed_lines"]):
            console.print("\033[F\033[K", end="")
        console.out("", end="")

        for line_out in lines:
            console.print(f"[dim]{line_out}[/dim]")
        state["printed_lines"] = len(lines)
        state["last_printed_block"] = lines.copy()

    def query_ollama_stream(self, prompt: str) -> str:
        """
        Streams expansions from a local Ollama server. 
        If unavailable, returns "[]".
        """
        url = "http://localhost:11434/api/generate"
        headers = {"Content-Type": "application/json"}
        data = {"model": self.model, "prompt": prompt, "stream": True}

        if not self.check_model_availability():
            console.print(f"[error]Model '{self.model}' not found in Ollama's environment.[/error]")
            console.print(f"[warning]Try 'ollama pull {self.model}' or adapt to your LLM backend.[/warning]")
            return "[]"

        try:
            resp = requests.post(url, headers=headers, json=data, stream=True)
            resp.raise_for_status()

            full_response = ""
            in_think_mode = False
            think_buffer = ""
            thinking_state = {
                "printed_lines": 0,
                "printed_brain_emoji": False,
                "last_printed_block": [],
            }

            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk_data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                chunk = chunk_data.get("response", "")
                full_response += chunk

                idx = 0
                while idx < len(chunk):
                    if not in_think_mode:
                        st_tag = chunk.find("<think>", idx)
                        if st_tag == -1:
                            break
                        idx = st_tag + len("<think>")
                        in_think_mode = True
                        thinking_state["printed_brain_emoji"] = False
                        thinking_state["last_printed_block"] = []
                    else:
                        end_tag = chunk.find("</think>", idx)
                        if end_tag == -1:
                            think_buffer += chunk[idx:]
                            self._update_thinking_block(think_buffer, thinking_state)
                            break
                        else:
                            think_buffer += chunk[idx:end_tag]
                            self._update_thinking_block(think_buffer, thinking_state)
                            in_think_mode = False
                            think_buffer = ""
                            idx = end_tag + len("</think>")

                if chunk_data.get("done", False):
                    break
            return self.strip_thinking_tags(full_response)

        except KeyboardInterrupt:
            raise
        except Exception as e:
            console.print(f"[error]Error streaming expansions from Ollama: {e}[/error]")
            return "[]"

    def get_related_concepts(self, concept: str, path: Optional[List[str]] = None) -> List[str]:
        """
        BFS expansions from a concept. Query LLM for ~8-10 expansions in JSON array.
        """
        if concept in self.seen_concepts:
            return []
        self.seen_concepts.add(concept)
        self.current_concept = concept
        full_path = (path or []) + [concept]

        prompt = textwrap.dedent(f"""
            You are an advanced BFS agent. Starting with "{concept}",
            produce 8-10 related concepts (1-5 words each) as a JSON array, e.g. ["C1","C2",...].
            Aim for diversity across fields, avoid duplicates, keep it intriguing.

            Path so far: {' -> '.join(full_path)}
        """).strip()

        console.print(f"\n[cyan]⚡ Expanding from concept:[/cyan] [yellow]{concept}[/yellow]")
        if path:
            console.print(f"[cyan]Path:[/cyan] [magenta]{' -> '.join(path)} -> {concept}[/magenta]")

        raw = self.query_ollama_stream(prompt)
        try:
            if "[" in raw and "]" in raw:
                chunk_start = raw.find("[")
                chunk_end = raw.rfind("]") + 1
                expansions_txt = raw[chunk_start:chunk_end]
                expansions = json.loads(expansions_txt)

                filtered = []
                for rc in expansions:
                    rc = rc.strip()
                    if not rc:
                        continue
                    # Avoid duplicates
                    if rc.lower() in (s.lower() for s in self.seen_concepts):
                        console.print(f"[error]✗ Duplicate concept: {rc}[/error]")
                        continue
                    # If overly long, truncate
                    if len(rc) > self.term_width // 3:
                        rc = rc[: self.term_width // 3 - 3] + "..."
                    filtered.append(rc)
                console.print(f"[success]✓ Found {len(filtered)} expansions[/success]")
                return filtered
            else:
                console.print(f"[error]✗ Invalid JSON in LLM response[/error]")
                console.print(f"[warning]Full response:\n{raw}[/warning]")
                return []
        except Exception as ex:
            console.print(f"[error]✗ Error parsing expansions: {ex}[/error]")
            console.print(f"[warning]Raw response:\n{raw}[/warning]")
            return []

    def _diversity_score(self, concept: str, existing: set) -> float:
        score = 0
        for e in existing:
            shared = set(concept.lower().split()) & set(e.lower().split())
            if not shared:
                score += 1
        return score

    def _ascii_tree(self, node: str, prefix="", is_last=True, visited=None) -> str:
        if visited is None:
            visited = set()
        if node in visited:
            return f"{prefix}└── (...)\n"
        visited.add(node)

        # Color the node
        if node == self.current_concept:
            display_node = f"[reverse magenta]{node}[/reverse magenta]"
        elif node == self.last_added:
            display_node = f"[reverse green]{node}[/reverse green]"
        else:
            display_node = f"[green]{node}[/green]"

        connector = "└── " if is_last else "├── "
        out = f"{prefix}{connector}{display_node}\n"
        successors = list(self.graph.successors(node))
        if not successors:
            return out
        next_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(successors):
            last_child = (i == len(successors) - 1)
            out += self._ascii_tree(child, next_prefix, last_child, visited)
        return out

    def update_live_tree(self, focus_node=None):
        now = time.time()
        if now - self.last_tree_update_time < self.MIN_UPDATE_INTERVAL:
            return
        self.last_tree_update_time = now
        self.term_width, self.term_height = shutil.get_terminal_size((80, 24))
        clear_terminal()

        # Banner
        header_lines = [
            f"[blue]{'=' * min(65, self.term_width)}[/blue]",
            "[bold yellow]🔱∞  AGENTIC ALPHA EXPLORER (INFINITE MODE)  ∞🔱[/bold yellow]",
            f"[blue]{'=' * min(65, self.term_width)}[/blue]\n"
        ]
        for line in header_lines:
            console.print(line, highlight=False)

        # Find root
        roots = [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]
        if not roots:
            console.print("[red]No root node found![/red]")
            return

        root = roots[0]
        ascii_str = self._ascii_tree(root)
        console.print(ascii_str, highlight=False)
        console.print(f"\n[yellow]{'=' * min(65, self.term_width)}[/yellow]")
        console.print(f"[cyan]Total Concepts: {len(self.graph.nodes)} | Connections: {len(self.graph.edges)}[/cyan]")
        if self.current_concept:
            console.print(f"[white]Expanding: [yellow]{self.current_concept}[/yellow][/white]")

    def build_concept_web_infinite(self, root_concept: str, diversity_bias: float = DEFAULT_DIVERSITY_BIAS):
        self.graph.add_node(root_concept)
        queue = deque([(root_concept, [])])

        try:
            while True:
                concept, path = queue.popleft()
                self.update_live_tree(focus_node=concept)
                expansions = self.get_related_concepts(concept, path)

                if diversity_bias > 0 and expansions and random.random() < diversity_bias:
                    expansions.sort(key=lambda x: self._diversity_score(x, self.seen_concepts))

                for c in expansions:
                    if c not in self.graph:
                        self.graph.add_node(c)
                    self.graph.add_edge(concept, c)
                    self.last_added = c
                    new_path = path + [concept]
                    queue.append((c, new_path))
                    self.update_live_tree(focus_node=c)
                    time.sleep(SLEEP_DURATION)

                time.sleep(SLEEP_DURATION)

        except KeyboardInterrupt:
            console.print(f"\n[yellow]Exploration ended by user (Ctrl+C).[/yellow]")
            return
        finally:
            self.current_concept = None
            self.last_added = None
            self.update_live_tree()
            console.print(f"\n[green]✨ BFS expansions ended (interrupt).[/green]")

    def build_concept_web_agentic_infinite(self, root_concept: str, diversity_bias: float = DEFAULT_DIVERSITY_BIAS):
        from collections import deque
        from time import sleep

        console.print(f"\n[magenta]==== AGENTIC TREE SEARCH MODE (INFINITE) ====[/magenta]")
        cfg = {
            "agent": {
                "num_workers": 1,
                "search": {"num_drafts": 1},
                "steps": 10,
            },
            "exec": {
                "timeout": 300,
            },
        }

        journal = Journal()
        root_node = Node(plan=f"Root concept: {root_concept}", is_seed_node=True)
        journal.append(root_node)
        self.graph.add_node(root_concept)
        queue = deque([(root_node, [])])

        with ParallelAgent(task_desc="Concept BFS", cfg=cfg, journal=journal) as agent:
            try:
                while True:
                    node, path = queue.popleft()
                    concept = node.plan.replace("Root concept:", "").strip()
                    self.update_live_tree(focus_node=concept)

                    console.print(f"\n[cyan]Agentic BFS exploring: [white]{concept}[/white][/cyan]")
                    expansions = self.get_related_concepts(concept, path)
                    if diversity_bias > 0 and expansions and random.random() < diversity_bias:
                        expansions.sort(key=lambda x: self._diversity_score(x, self.seen_concepts))

                    for c in expansions:
                        if c not in self.graph:
                            self.graph.add_node(c)
                        self.graph.add_edge(concept, c)
                        self.last_added = c
                        child_node = Node(plan=c, parent=node)
                        journal.append(child_node)
                        queue.append((child_node, path + [concept]))
                        self.update_live_tree(focus_node=c)
                        sleep(SLEEP_DURATION)

                    sleep(SLEEP_DURATION)

            except KeyboardInterrupt:
                console.print(f"\n[yellow]Exploration ended by user (Ctrl+C).[/yellow]")
            finally:
                self.current_concept = None
                self.last_added = None
                self.update_live_tree()
                console.print(f"\n[green]✨ Agentic BFS expansions ended (interrupt).[/green]")

    def export_ascii_tree(self, output_file: str = "concept_web.txt"):
        roots = [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]
        if not roots:
            console.print("[red]No root node found; nothing to export.[/red]")
            return

        def _plain_ascii_tree(node, prefix="", is_last=True, visited=None):
            if visited is None:
                visited = set()
            if node in visited:
                return f"{prefix}{'└── ' if is_last else '├── '}{node} (...)\n"
            visited.add(node)
            out_text = f"{prefix}{'└── ' if is_last else '├── '}{node}\n"
            children = list(self.graph.successors(node))
            if not children:
                return out_text

            next_prefix = prefix + ("    " if is_last else "│   ")
            for i, c in enumerate(children):
                last_c = (i == len(children) - 1)
                out_text += _plain_ascii_tree(c, next_prefix, last_c, visited)
            return out_text

        root = roots[0]
        ascii_str = _plain_ascii_tree(root)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(ascii_str)
        console.print(f"[green]📝 ASCII tree exported to '{output_file}'[/green]")


# ------------------------------------------------------------------------------
# CLI Argument Parsing & Main
# ------------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Discover infinite BFS expansions of a root concept until Ctrl+C."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Name of the LLM model (e.g. 'llama2'). Must exist locally or adapt the script to your LLM."
    )
    parser.add_argument(
        "--root",
        default=DEFAULT_ROOT_CONCEPT,
        help="Root concept for BFS expansions."
    )
    parser.add_argument(
        "--diversity",
        type=float,
        default=DEFAULT_DIVERSITY_BIAS,
        help="Probability to reorder expansions by diversity (0 <= x <= 1)."
    )
    parser.add_argument(
        "--depth",
        "--max-depth",
        dest="depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        help="Ignored in infinite mode; retained for legacy usage."
    )
    parser.add_argument(
        "--agentic",
        action="store_true",
        help="Enable agentic BFS expansions (infinite) instead of classic BFS."
    )
    return parser.parse_args()


def main():
    clear_terminal()

    # Intro Banner
    banner_lines = [
        "[green]" + "=" * 70 + "[/green]",
        "[bold yellow]🪐🔮  AGENTIC ALPHA EXPLORER (INFINITE) - ASI SUPERINTELLIGENCE ERA  🔮🪐[/bold yellow]",
        "[green]" + "=" * 70 + "[/green]\n",
        "[magenta]Press Ctrl+C at any time to halt expansions.[/magenta]\n"
    ]
    for line in banner_lines:
        console.print(line)

    args = parse_arguments()
    console.print(f"[yellow]Root Concept:    [/yellow][white]{args.root}[/white]")
    console.print(f"[yellow]Model:           [/yellow][white]{args.model}[/white]")
    console.print(f"[yellow]Diversity:       [/yellow][white]{args.diversity}[/white]")
    console.print(f"[yellow]Depth (ignored): [/yellow][white]{args.depth}[/white]")
    if args.agentic:
        console.print(f"[yellow]Mode:            [/yellow][white]Agentic BFS (infinite)[/white]")
    else:
        console.print(f"[yellow]Mode:            [/yellow][white]Classic BFS (infinite)[/white]")

    explorer = ConceptExplorer(model=args.model)

    # Check if the model is available if using Ollama
    if not explorer.check_model_availability():
        console.print(f"[error]Error: Model '{args.model}' not available in Ollama or your LLM backend.[/error]")
        available = explorer.get_available_models()
        if available:
            console.print("[green]Available models:[/green]")
            for i, am in enumerate(available, 1):
                console.print(f"[cyan]{i}. {am}[/cyan]")
        else:
            console.print("[warning]No models found. You may need to pull or adapt the script to your LLM backend.[/warning]")
        sys.exit(1)

    try:
        if args.agentic:
            explorer.build_concept_web_agentic_infinite(args.root, diversity_bias=args.diversity)
        else:
            explorer.build_concept_web_infinite(args.root, diversity_bias=args.diversity)

        out_file = f"{args.root.lower().replace(' ', '_')}_concept_web.txt"
        explorer.export_ascii_tree(out_file)
        console.print(
            f"\n[green]✨ Exploration finished! {len(explorer.graph.nodes)} concepts, "
            f"{len(explorer.graph.edges)} connections. Results in '{out_file}'.[/green]"
        )

    except KeyboardInterrupt:
        console.print(f"\n[yellow]Exploration interrupted by user (Ctrl+C).[/yellow]")
        out_file = f"{args.root.lower().replace(' ', '_')}_concept_web.txt"
        explorer.export_ascii_tree(out_file)
        console.print(
            f"[green]Partial expansions saved with {len(explorer.graph.nodes)} concepts. "
            f"ASCII tree in '{out_file}'.[/green]"
        )

    except Exception as e:
        console.print(f"\n[error]An unexpected error occurred: {e}[/error]")
        traceback.print_exc()
        sys.exit(1)

    finally:
        console.print("\n[bold magenta]🪐 Session ended. If terminal is misaligned, type 'reset'.[/bold magenta]\n")


if __name__ == "__main__":
    main()

