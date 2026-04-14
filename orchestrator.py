import time
from tools import handle_filesystem, handle_llm_generate, handle_open_app, handle_screenshot

# The registry: maps action types to handler functions.
# Adding a new capability = one new entry here + one handler function in tools.py.
REGISTRY = {
    "create_file": handle_filesystem,
    "write_code": handle_llm_generate,
    "summarize": handle_llm_generate,
    "general_chat": handle_llm_generate,
    "open_app": handle_open_app,
    "screenshot": handle_screenshot,
}

# Actions that must run before others when both are present.
# Key runs before value. This is where sequencing logic lives.
ORDERING_RULES = {
    "write_code": "create_file",  # generate code before writing to file
}


def execute(actions: list) -> dict:
    """
    Takes the action list from the classifier, executes each action
    through its handler, returns the full execution context.
    """
    context = {}

    # Sort actions based on ordering rules
    actions = _sort_actions(actions)

    results = []
    for action in actions:
        action_type = action.get("type", "general_chat")
        params = action.get("params", {})

        handler = REGISTRY.get(action_type)
        if not handler:
            results.append({
                "action": action_type,
                "status": "error",
                "message": f"Unknown action type: {action_type}"
            })
            continue

        # Tell the handler what kind of action triggered it
        # (handle_llm_generate needs this to pick the right prompt)
        context["current_action_type"] = action_type

        result = handler(params, context)
        result["action"] = action_type
        results.append(result)

        # Async side-effect handling: if this action triggered an app/URL to open,
        # wait a moment before the next action so screenshots capture the loaded state.
        if action_type == "open_app" and result.get("status") == "opened":
            # Check if a screenshot follows in the action list
            remaining = actions[actions.index(action) + 1:]
            if any(a.get("type") == "screenshot" for a in remaining):
                time.sleep(3)  # give the browser/app time to render

    context["results"] = results
    return context


def _sort_actions(actions: list) -> list:
    """
    Reorder actions based on dependency rules.
    If write_code and create_file both exist, write_code goes first.
    """
    action_types = [a.get("type") for a in actions]
    sorted_actions = list(actions)

    for before, after in ORDERING_RULES.items():
        if before in action_types and after in action_types:
            before_idx = action_types.index(before)
            after_idx = action_types.index(after)
            if before_idx > after_idx:
                # Swap them
                sorted_actions[before_idx], sorted_actions[after_idx] = (
                    sorted_actions[after_idx], sorted_actions[before_idx]
                )

    return sorted_actions
