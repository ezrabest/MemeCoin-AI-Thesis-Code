from pathlib import Path
import re

path = Path("app/clean_forward/symbol_rehydration.py")
text = path.read_text(encoding="utf-8")

backup = path.with_suffix(".py.bak_before_test_guard_fix")
backup.write_text(text, encoding="utf-8")

if "explicit_validator_supplied = validator is not None" not in text:
    old = '''    if validator is None:
        from app.ae13b_product.dexscreener_pair_verify import validate_dexscreener_pair
'''
    new = '''    explicit_validator_supplied = validator is not None
    if validator is None:
        from app.ae13b_product.dexscreener_pair_verify import validate_dexscreener_pair
'''
    if old not in text:
        raise SystemExit("Could not find validator defaulting block. File shape changed.")
    text = text.replace(old, new, 1)

old_cond = '''    if not (cell(payload.get("provider_base_token_symbol")) and cell(payload.get("provider_quote_token_symbol"))):
        payload, resolver_meta = _lookup_pair_payload_multistage(row, chain=chain, segment=segment)
'''
new_cond = '''    if (
        not explicit_validator_supplied
        and not (cell(payload.get("provider_base_token_symbol")) and cell(payload.get("provider_quote_token_symbol")))
    ):
        payload, resolver_meta = _lookup_pair_payload_multistage(row, chain=chain, segment=segment)
'''

if old_cond not in text:
    raise SystemExit("Could not find multi-stage fallback condition. File shape changed.")

text = text.replace(old_cond, new_cond, 1)

path.write_text(text, encoding="utf-8")
print(f"Updated {path}")
print(f"Backup written to {backup}")
