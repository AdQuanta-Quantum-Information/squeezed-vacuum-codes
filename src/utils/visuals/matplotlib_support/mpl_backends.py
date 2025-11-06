import matplotlib as mpl
import os
import subprocess


def _detect_backend() -> str:
    """Automatically detect the appropriate matplotlib backend.
    returns:
        str: The name of the backend to use.
              Either 'TkAgg' for GUI environments or 'Agg' for headless environments.
    
    Raises:
        RuntimeError: If backend cannot be determined reliably
    """
    
    # Check if we're in a known headless environment
    if (os.environ.get('CODESPACES') or 
        os.environ.get('CI') or 
        os.environ.get('GITHUB_ACTIONS')):
        return 'Agg'
    

    
    # Try TkAgg if we think we have a display
    try:
        import tkinter
        # Test if tkinter can actually create a window
        root = tkinter.Tk()
        root.withdraw()  # Hide the window
        root.destroy()
        return 'TkAgg'
    except (ImportError, tkinter.TclError):
        # If tkinter fails or can't connect to display
        raise RuntimeError(
            "Cannot determine appropriate matplotlib backend. "
            "GUI environment detected but tkinter is not available or cannot connect to display. "
            "Please manually set the backend using mpl.use('backend_name')"
        )



if __name__ == "__main__":
    ## select the proper backend for rendering figures: 
    try:
        backend = _detect_backend()
        mpl.use(backend)
        print(f"Using matplotlib backend: {backend}")
    except ImportError as e:
        import warnings
        warnings.warn(str(e))
    except RuntimeError as e:
        raise e
