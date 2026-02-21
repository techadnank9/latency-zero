from typing import Dict, List


def create_event(title: str, time: str) -> str:
    return f"Event created: '{title}' at {time}."


def send_message(recipient: str, message: str) -> str:
    return f"Message sent to {recipient}: '{message}'."


def add_task(task: str) -> str:
    return f"Task added: '{task}'."


def set_mode(mode: str) -> str:
    return f"Mode set to '{mode}'."


TOOL_SCHEMAS: List[Dict] = [
    {
        "name": "create_event",
        "description": "Create a calendar event with title and time.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "time": {"type": "string"},
            },
            "required": ["title", "time"],
        },
    },
    {
        "name": "send_message",
        "description": "Send a short message to a recipient.",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string"},
                "message": {"type": "string"},
            },
            "required": ["recipient", "message"],
        },
    },
    {
        "name": "add_task",
        "description": "Add a task to a to-do list.",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
            },
            "required": ["task"],
        },
    },
    {
        "name": "set_mode",
        "description": "Set application mode.",
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {"type": "string"},
            },
            "required": ["mode"],
        },
    },
]
