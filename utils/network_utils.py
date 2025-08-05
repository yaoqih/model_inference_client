"""
Network utility functions.
"""
import socket

def find_free_port() -> int:
    """
    Finds a free port on the host machine.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]