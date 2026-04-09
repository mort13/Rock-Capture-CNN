#!/usr/bin/env python3
"""
Test script to verify integer casting for JSON values.
Tests the casting logic for mass, resistance, quality, and _int/_dec fields.
"""


def _should_cast_as_int(field_name: str) -> bool:
    """
    Determine if a field should be cast to integer in JSON output.
    
    Cast to int for:
    - mass, resistance, quality (specific fields)
    - Any field ending with _int or _dec
    """
    if field_name in ("mass", "resistance", "quality"):
        return True
    if field_name.endswith("_int") or field_name.endswith("_dec"):
        return True
    return False


def _cast_value_if_needed(value: str | int | None, field_name: str) -> str | int | None:
    """
    Cast value to int if the field requires it, otherwise return as-is.
    """
    if value is None or value == "":
        return value
    if not _should_cast_as_int(field_name):
        return value
    try:
        return int(value) if isinstance(value, str) else value
    except (ValueError, TypeError):
        return value


def _apply_int_casting_to_dict(data: dict, parent_key: str = "") -> None:
    """
    Recursively walk through a dictionary and cast string values to integers
    for fields that should be integers.
    """
    for key, value in data.items():
        if isinstance(value, dict) and "value" in value:
            # This is a value/confidence pair
            if _should_cast_as_int(key):
                value["value"] = _cast_value_if_needed(value["value"], key)
        elif isinstance(value, dict):
            # Recurse into nested dicts
            _apply_int_casting_to_dict(value, key)
        elif isinstance(value, list):
            # Handle arrays (e.g., composition array)
            for item in value:
                if isinstance(item, dict):
                    _apply_int_casting_to_dict(item, key)


def test_should_cast_as_int():
    """Test the field name checking logic."""
    # Fields that should cast to int
    assert _should_cast_as_int("mass") == True
    assert _should_cast_as_int("resistance") == True
    assert _should_cast_as_int("quality") == True
    assert _should_cast_as_int("amount_int") == True
    assert _should_cast_as_int("amount_dec") == True
    assert _should_cast_as_int("instability_int") == True
    assert _should_cast_as_int("volume_dec") == True
    
    # Fields that should NOT cast to int
    assert _should_cast_as_int("name") == False
    assert _should_cast_as_int("deposit_name") == False
    assert _should_cast_as_int("confidence") == False
    assert _should_cast_as_int("amount") == False  # no _int or _dec suffix
    
    print("✓ test_should_cast_as_int passed")


def test_cast_value_if_needed():
    """Test the value casting logic."""
    # Should cast these to int
    assert _cast_value_if_needed("11826", "mass") == 11826
    assert _cast_value_if_needed("0", "resistance") == 0
    assert _cast_value_if_needed("434", "quality") == 434
    assert _cast_value_if_needed("3", "amount_int") == 3
    assert _cast_value_if_needed("40", "amount_dec") == 40
    
    # Already int type
    assert _cast_value_if_needed(11826, "mass") == 11826
    
    # Should NOT cast these
    assert _cast_value_if_needed("silicon", "name") == "silicon"
    assert _cast_value_if_needed("gold", "deposit_name") == "gold"
    
    # Edge cases
    assert _cast_value_if_needed(None, "mass") is None
    assert _cast_value_if_needed("", "mass") == ""
    assert _cast_value_if_needed("invalid", "mass") == "invalid"  # Can't convert
    
    print("✓ test_cast_value_if_needed passed")


def test_apply_int_casting_to_dict():
    """Test the recursive dictionary casting."""
    # Test structured data format
    data = {
        "scan": {
            "deposit_name": {"value": "silicon", "confidence": 0.799},
            "mass": {"value": "11826", "confidence": 0.994},
            "resistance": {"value": "0", "confidence": 0.999},
            "composition": [
                {
                    "name": {"value": "borase", "confidence": 0.753},
                    "amount_int": {"value": "3", "confidence": 0.999},
                    "amount_dec": {"value": "40", "confidence": 0.997},
                    "quality": {"value": "434", "confidence": 1.0},
                },
                {
                    "name": {"value": "silicon", "confidence": 0.995},
                    "amount_int": {"value": "4", "confidence": 0.999},
                    "amount_dec": {"value": "36", "confidence": 0.997},
                    "quality": {"value": "528", "confidence": 1.0},
                }
            ]
        }
    }
    
    _apply_int_casting_to_dict(data)
    
    # Check that type conversions happened
    assert data["scan"]["mass"]["value"] == 11826, f"Expected 11826, got {data['scan']['mass']['value']}"
    assert isinstance(data["scan"]["mass"]["value"], int), "mass should be int"
    
    assert data["scan"]["resistance"]["value"] == 0, f"Expected 0, got {data['scan']['resistance']['value']}"
    assert isinstance(data["scan"]["resistance"]["value"], int), "resistance should be int"
    
    # Check that strings are preserved where needed
    assert data["scan"]["deposit_name"]["value"] == "silicon", "deposit_name should remain string"
    assert isinstance(data["scan"]["deposit_name"]["value"], str), "deposit_name should be string"
    
    # Check array items
    assert data["scan"]["composition"][0]["amount_int"]["value"] == 3
    assert isinstance(data["scan"]["composition"][0]["amount_int"]["value"], int)
    
    assert data["scan"]["composition"][0]["name"]["value"] == "borase"
    assert isinstance(data["scan"]["composition"][0]["name"]["value"], str)
    
    assert data["scan"]["composition"][0]["quality"]["value"] == 434
    assert isinstance(data["scan"]["composition"][0]["quality"]["value"], int)
    
    print("✓ test_apply_int_casting_to_dict passed")


def test_legacy_flat_format():
    """Test casting in legacy flat format."""
    data = {
        "values": {
            "mole_profile": {
                "deposit_name": {"value": "gold", "confidence": 0.85},
                "mass": {"value": "5000", "confidence": 0.95},
                "resistance": {"value": "10", "confidence": 0.92},
                "word_model": {"value": "gold", "confidence": 0.98},
            }
        }
    }
    
    _apply_int_casting_to_dict(data)
    
    # Check conversions
    assert data["values"]["mole_profile"]["mass"]["value"] == 5000
    assert isinstance(data["values"]["mole_profile"]["mass"]["value"], int)
    
    assert data["values"]["mole_profile"]["resistance"]["value"] == 10
    assert isinstance(data["values"]["mole_profile"]["resistance"]["value"], int)
    
    # Check string preservation
    assert data["values"]["mole_profile"]["deposit_name"]["value"] == "gold"
    assert isinstance(data["values"]["mole_profile"]["deposit_name"]["value"], str)
    
    print("✓ test_legacy_flat_format passed")


if __name__ == "__main__":
    test_should_cast_as_int()
    test_cast_value_if_needed()
    test_apply_int_casting_to_dict()
    test_legacy_flat_format()
    print("\n✅ All tests passed!")
