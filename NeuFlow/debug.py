import inspect
import os

# Custom print function fprint to handle multiple arguments
def fprint(*args):
    # Retrieve the caller's stack frame
    frame = inspect.stack()[1]
    filename = os.path.relpath(frame.filename)
    line_info = f"[{filename}:{frame.lineno}]"
    colored_line_info = f"\033[32m{line_info}\033[0m"  # 32 is the ANSI code for green
    
    # Convert all arguments to string and join them
    message = " ".join(map(str, args))
    
    # Print the formatted message
    print(f"{colored_line_info} {message}")
