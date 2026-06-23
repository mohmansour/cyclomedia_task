import pytest
import time
from typing import Dict, Any
from src.pipeline.dag import Pipeline, Task

class MockSuccessTask(Task):
    def __init__(self, name: str, dependencies=None):
        super().__init__(name, dependencies)
        self.runs = 0

    def run(self, context: Dict[str, Any]) -> Any:
        self.runs += 1
        return f"{self.name}_output"

class MockFailTask(Task):
    def __init__(self, name: str, retries: int = 3):
        super().__init__(name, retries=retries, retry_delay=0.1)
        self.runs = 0

    def run(self, context: Dict[str, Any]) -> Any:
        self.runs += 1
        raise ValueError("Simulated Task Failure")

def test_topological_sort_and_context_passing():
    """Verifies that tasks execute in topological order and context is correctly populated."""
    pipeline = Pipeline()
    task_a = MockSuccessTask("TaskA")
    task_b = MockSuccessTask("TaskB", dependencies=["TaskA"])
    task_c = MockSuccessTask("TaskC", dependencies=["TaskA"])
    task_d = MockSuccessTask("TaskD", dependencies=["TaskB", "TaskC"])
    
    pipeline.add_task(task_a)
    pipeline.add_task(task_b)
    pipeline.add_task(task_c)
    pipeline.add_task(task_d)
    
    execution_order = pipeline._topological_sort()
    
    # TaskA must run first
    assert execution_order[0] == "TaskA"
    # TaskD must run last
    assert execution_order[-1] == "TaskD"
    
    context = pipeline.run()
    
    # All tasks should execute exactly once
    assert task_a.runs == 1
    assert task_b.runs == 1
    assert task_c.runs == 1
    assert task_d.runs == 1
    
    # Context must collect outputs
    assert context["TaskA"] == "TaskA_output"
    assert context["TaskB"] == "TaskB_output"
    assert context["TaskC"] == "TaskC_output"
    assert context["TaskD"] == "TaskD_output"

def test_unregistered_dependency():
    """Verifies that depending on a non-existent task raises a ValueError."""
    pipeline = Pipeline()
    task_a = MockSuccessTask("TaskA", dependencies=["MissingTask"])
    pipeline.add_task(task_a)
    
    with pytest.raises(ValueError, match="depends on unregistered task"):
        pipeline.run()

def test_dependency_cycle():
    """Verifies that circular dependencies are detected and raise a ValueError."""
    pipeline = Pipeline()
    task_a = MockSuccessTask("TaskA", dependencies=["TaskB"])
    task_b = MockSuccessTask("TaskB", dependencies=["TaskA"])
    
    pipeline.add_task(task_a)
    pipeline.add_task(task_b)
    
    with pytest.raises(ValueError, match="Cycle detected in pipeline DAG"):
        pipeline.run()

def test_task_retries_and_failure():
    """Verifies that a failing task is retried up to its limit and raises an exception on final fail."""
    pipeline = Pipeline()
    fail_task = MockFailTask("FailTask", retries=3)
    pipeline.add_task(fail_task)
    
    start_time = time.time()
    with pytest.raises(ValueError, match="Simulated Task Failure"):
        pipeline.run()
    duration = time.time() - start_time
    
    # Should run exactly 3 times
    assert fail_task.runs == 3
    # Duration should be at least 0.2 seconds (due to 2 retry delays of 0.1s each)
    assert duration >= 0.2
