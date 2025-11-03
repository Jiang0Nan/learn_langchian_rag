import os

PROJECT_BASE = None

def get_project_base_directory(*args):
    global PROJECT_BASE
    if PROJECT_BASE is None:
        PROJECT_BASE = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                os.pardir,
                os.pardir,
            )
        )

    if args:
        return os.path.join(PROJECT_BASE, *args)
    return PROJECT_BASE


from . import rag_tokenizer
# def num_tokens_from_string(string: str) -> int:
#     """Returns the number of tokens in a text string."""
#     try:
#         return len(encoder.encode(string))
#     except Exception:
#         return 0
