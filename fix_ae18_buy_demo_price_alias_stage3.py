from pathlib import Path
import py_compile

path = Path("app/api.py")
text = path.read_text(encoding="utf-8")
path.with_suffix(".py.bak_ae18_buy_demo_price_alias_stage3").write_text(text, encoding="utf-8")

old = '''            "canonical_market_identity": canonical,
            "canonical_market_identity_type": "PROVIDER_URL",
            "provider_pair_url_exact": row.get("provider_pair_url_exact") or canonical,
            "provider_pair_url_final_segment_exact": row.get(
                "provider_pair_url_final_segment_exact"
            ),
            "open_chart_url": row.get("open_chart_url") or canonical,
            # derived helper only — never the canonical key
            "pair_address": row.get("pair_address_derived"),
            "pair_address_derived": row.get("pair_address_derived"),
            "token_contract_address": row.get("provider_base_token_address")
            or row.get("base_token_address_derived"),
            "price_usd": price_f,
            "price_source": "market_canonical_url",
            "price_timestamp": loaded.get("loaded_at"),
            "liquidity_usd": row.get("liquidity_usd"),
            "volume_24h": row.get("volume_h24"),
            "price_change_m5": row.get("price_change_m5"),
'''

new = '''            "canonical_market_identity": canonical,
            "canonical_market_identity_type": "PROVIDER_URL",
            "provider_pair_url_exact": row.get("provider_pair_url_exact") or canonical,
            "normalized_provider_pair_url_key": row.get("normalized_provider_pair_url_key"),
            "provider_pair_url_final_segment_exact": row.get(
                "provider_pair_url_final_segment_exact"
            ),
            "open_chart_url": row.get("open_chart_url") or canonical,

            # URL-first paper/demo execution identity.
            # pair_address remains a derived/helper field only, never canonical.
            "instrument_id": f"clean_forward:{row.get('chain') or ''}:{canonical}",
            "execution_instrument_id": f"clean_forward:{row.get('chain') or ''}:{canonical}",
            "instrument_source": "clean_forward_market_feed",
            "candidate_source": "clean_forward_market_feed",
            "mark_price_lookup_key": canonical,

            # derived helper only — never the canonical key
            "pair_address": row.get("pair_address_derived"),
            "pair_address_derived": row.get("pair_address_derived"),
            "token_contract_address": row.get("provider_base_token_address")
            or row.get("base_token_address_derived"),
            "base_token_address": row.get("provider_base_token_address"),
            "quote_token_address": row.get("provider_quote_token_address"),
            "provider_base_token_address": row.get("provider_base_token_address"),
            "provider_quote_token_address": row.get("provider_quote_token_address"),

            # Price aliases required by legacy paper/demo risk guards.
            # These all represent the same cached current mark price from the runtime index.
            "price_usd": price_f,
            "latest_price": price_f,
            "price": price_f,
            "market_price_usd": price_f,
            "price_source": "market_canonical_url",
            "entry_price_source": "market_canonical_url",
            "price_timestamp": loaded.get("loaded_at"),
            "last_seen_at": loaded.get("loaded_at"),
            "price_updated_at": loaded.get("loaded_at"),

            # Liquidity/volume aliases required by legacy risk guards.
            "liquidity_usd": row.get("liquidity_usd"),
            "latest_liquidity": row.get("liquidity_usd"),
            "liquidity_at_entry": row.get("liquidity_usd"),
            "volume_24h": row.get("volume_h24"),
            "latest_volume_24h": row.get("volume_h24"),
            "price_change_m5": row.get("price_change_m5"),
'''

if old not in text:
    raise SystemExit("FAILED: could not find demo buy candidate coin block. api.py shape changed.")

text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")

py_compile.compile(str(path), doraise=True)
print("OK syntax: app/api.py")
print("AE18 stage3 buy-demo price alias patch applied.")
