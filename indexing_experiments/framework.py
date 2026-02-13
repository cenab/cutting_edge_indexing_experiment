try:
    from .llama_framework import *  # noqa: F401,F403
    from .llama_framework import _main as _llama_main
except ImportError:
    from llama_framework import *  # type: ignore # noqa: F401,F403
    from llama_framework import _main as _llama_main  # type: ignore


if __name__ == "__main__":
    _llama_main()
