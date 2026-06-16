import logging
import time
from typing import Dict, List, Any, Set

logger = logging.getLogger("PipelineEngine")

class Task:
    """
    Represents a modular unit of work in the pipeline DAG.
    """
    def __init__(self, name: str, dependencies: List[str] = None, retries: int = 3, retry_delay: float = 2.0):
        self.name = name
        self.dependencies = dependencies or []
        self.retries = retries
        self.retry_delay = retry_delay

    def run(self, context: Dict[str, Any]) -> Any:
        """
        Executes the task. Must be overridden by subclasses.
        Returns data to be added to the pipeline context under the task's name.
        """
        raise NotImplementedError("Tasks must implement the run method.")


class Pipeline:
    """
    Manages and executes a set of tasks as a Directed Acyclic Graph (DAG).
    """
    def __init__(self):
        self.tasks: Dict[str, Task] = {}

    def add_task(self, task: Task) -> 'Pipeline':
        if task.name in self.tasks:
            raise ValueError(f"Task '{task.name}' is already defined in the pipeline.")
        self.tasks[task.name] = task
        return self

    def _topological_sort(self) -> List[str]:
        """
        Sorts the tasks topologically to determine execution order.
        Detects cycles and missing dependencies.
        """
        adj: Dict[str, List[str]] = {name: [] for name in self.tasks}
        in_degree: Dict[str, int] = {name: 0 for name in self.tasks}

        for name, task in self.tasks.items():
            for dep in task.dependencies:
                if dep not in self.tasks:
                    raise ValueError(f"Task '{name}' depends on unregistered task '{dep}'.")
                adj[dep].append(name)
                in_degree[name] += 1

        # Queue of nodes with no dependencies (in-degree 0)
        queue = [name for name, deg in in_degree.items() if deg == 0]
        execution_order = []

        while queue:
            curr = queue.pop(0)
            execution_order.append(curr)
            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(execution_order) != len(self.tasks):
            # Find cycle elements
            remaining = [name for name, deg in in_degree.items() if deg > 0]
            raise ValueError(f"Cycle detected in pipeline DAG. Nodes involved: {remaining}")

        return execution_order

    def run(self, initial_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Runs all tasks in topological order.
        If a task fails, it is retried up to its specified retry limit.
        """
        context = initial_context or {}
        execution_order = self._topological_sort()
        
        logger.info(f"Starting pipeline execution with {len(execution_order)} tasks.")
        logger.info(f"Execution path: {' -> '.join(execution_order)}")

        for task_name in execution_order:
            task = self.tasks[task_name]
            success = False
            last_exception = None

            for attempt in range(1, task.retries + 1):
                logger.info(f"Executing task '{task_name}' (Attempt {attempt}/{task.retries})...")
                start_time = time.time()
                
                try:
                    output = task.run(context)
                    context[task_name] = output
                    success = True
                    duration = time.time() - start_time
                    logger.info(f"Task '{task_name}' completed successfully in {duration:.2f}s.")
                    break
                except Exception as e:
                    last_exception = e
                    duration = time.time() - start_time
                    logger.error(
                        f"Task '{task_name}' failed on attempt {attempt}/{task.retries} after {duration:.2f}s. "
                        f"Error: {str(e)}", 
                        exc_info=True
                    )
                    if attempt < task.retries:
                        logger.info(f"Waiting {task.retry_delay}s before retrying '{task_name}'...")
                        time.sleep(task.retry_delay)

            if not success:
                logger.critical(f"Pipeline execution aborted: Task '{task_name}' failed after {task.retries} attempts.")
                if last_exception:
                    raise last_exception
                raise RuntimeError(f"Task '{task_name}' failed to execute.")

        logger.info("Pipeline execution completed successfully.")
        return context
