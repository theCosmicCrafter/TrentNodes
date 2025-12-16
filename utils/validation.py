"""
Node validation and self-test utilities for TrentNodes.
"""
import os
import time
import traceback

from .banner import print_status


def validate_node_class(cls) -> list:
    """
    Validate that a class has all required ComfyUI node attributes.

    Args:
        cls: Node class to validate

    Returns:
        List of problem descriptions (empty if valid)
    """
    problems = []

    if not hasattr(cls, "FUNCTION"):
        problems.append("missing FUNCTION")

    if not hasattr(cls, "RETURN_TYPES"):
        problems.append("missing RETURN_TYPES")

    if not hasattr(cls, "INPUT_TYPES"):
        problems.append("missing INPUT_TYPES()")
    else:
        try:
            input_types = cls.INPUT_TYPES()
            if not isinstance(input_types, dict):
                problems.append("INPUT_TYPES() did not return a dict")
        except Exception as e:
            problems.append(f"INPUT_TYPES() error: {e}")

    if not hasattr(cls, "CATEGORY"):
        problems.append("missing CATEGORY (node won't show in a folder)")

    return problems


def validate_mappings(
    class_mappings: dict,
    display_mappings: dict
) -> list:
    """
    Validate NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS.

    Args:
        class_mappings: NODE_CLASS_MAPPINGS dict
        display_mappings: NODE_DISPLAY_NAME_MAPPINGS dict

    Returns:
        List of issue descriptions
    """
    issues = []

    # Validate each registered node class
    for name, cls in class_mappings.items():
        problems = validate_node_class(cls)
        if problems:
            issues.append(f"{name}: " + ", ".join(problems))

    # Check for display names without corresponding classes
    orphan_names = [k for k in display_mappings.keys() if k not in class_mappings]
    if orphan_names:
        issues.append(
            "DISPLAY_NAME keys without classes: " + ", ".join(orphan_names)
        )

    return issues


def run_self_test(class_mappings: dict, display_mappings: dict):
    """
    Run self-test on registered nodes and print results.

    Args:
        class_mappings: NODE_CLASS_MAPPINGS dict
        display_mappings: NODE_DISPLAY_NAME_MAPPINGS dict

    Raises:
        RuntimeError: If TRENT_STRICT=1 and issues are found
    """
    start = time.perf_counter()

    try:
        if not class_mappings:
            raise RuntimeError("No nodes registered in NODE_CLASS_MAPPINGS")

        issues = validate_mappings(class_mappings, display_mappings)
        elapsed = int((time.perf_counter() - start) * 1000)

        if not issues:
            status_msg = (
                f"Trent Nodes loaded OK - {len(class_mappings)} node(s) "
                f"ready ({elapsed} ms)."
            )
            print_status(True, status_msg)
        else:
            msg = (
                f"Trent Nodes loaded with {len(issues)} issue(s) "
                f"({elapsed} ms)."
            )
            details = "\n - " + "\n - ".join(issues)
            print_status(False, msg, details=details)

            if os.environ.get("TRENT_STRICT", "0") == "1":
                raise RuntimeError(msg + "\n" + "\n".join(issues))

    except Exception as e:
        tb = traceback.format_exc(limit=3)
        print_status(
            False,
            f"Trent Nodes failed to initialize: {e}",
            details=tb
        )
        if os.environ.get("TRENT_STRICT", "0") == "1":
            raise
