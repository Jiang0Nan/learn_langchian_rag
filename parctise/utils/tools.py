import bisect
import hashlib
import re
from typing import List

def charpos_to_page(char_pos: int, page_starts: List[int]) -> int:
    """
    给定字符位置 char_pos，和 page_starts 列表（每页起始字符索引），
    返回该位置属于哪个页（返回 page index 的列表下标），
    若 char_pos 恰好等于某页起始，则属于该页。
    """
    # bisect_right-1 可以得到 index
    idx = bisect.bisect_right(page_starts, char_pos) - 1
    if idx < 0:
        return 0
    return idx

def add_space_between_eng_zh(txt):
        # 在中文（汉字）与英文或数字混排时自动插入空格，
        # (ENG/ENG+NUM) + ZH
        txt = re.sub(r'([A-Za-z]+[0-9]+)([\u4e00-\u9fa5]+)', r'\1 \2', txt)
        # ENG + ZH
        txt = re.sub(r'([A-Za-z])([\u4e00-\u9fa5]+)', r'\1 \2', txt)
        # ZH + (ENG/ENG+NUM)
        txt = re.sub(r'([\u4e00-\u9fa5]+)([A-Za-z]+[0-9]+)', r'\1 \2', txt)
        txt = re.sub(r'([\u4e00-\u9fa5]+)([A-Za-z])', r'\1 \2', txt)
        return txt

def rmWWW(txt):
    """处理虚词"""
    patts = [
        (
            r"是*(什么样的|哪家|一下|那家|请问|啥样|咋样了|什么时候|何时|何地|何人|是否|是不是|多少|哪里|怎么|哪儿|怎么样|如何|哪些|是啥|啥是|啊|吗|呢|吧|咋|什么|有没有|呀|谁|哪位|哪个)是*",
            "",
        ),
        (r"(^| )(what|who|how|which|where|why)('re|'s)? ", " "),
        (
            r"(^| )('s|'re|is|are|were|was|do|does|did|don't|doesn't|didn't|has|have|be|there|you|me|your|my|mine|just|please|may|i|should|would|wouldn't|will|won't|done|go|for|with|so|the|a|an|by|i'm|it's|he's|she's|they|they're|you're|as|by|on|in|at|up|out|down|of|to|or|and|if) ",
            " ")
    ]
    otxt = txt
    for r, p in patts:
        txt = re.sub(r, p, txt, flags=re.IGNORECASE)
    if not txt:
        txt = otxt
    return txt


def hash_str2int(line:str, mod: int=10 ** 8) -> int:
    return int(hashlib.sha1(line.encode("utf-8")).hexdigest(), 16) % mod