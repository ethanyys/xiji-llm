import re
import hashlib


def to_md5_str(org_str, code="utf-8"):
    return hashlib.md5(org_str.encode(code)).hexdigest()

