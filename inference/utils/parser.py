import json, json5
from typing import Dict, Any

def parser(raw: str) -> Dict[str, Any]:
    if raw == "":
        return None
    raw_content = raw.split("```json")[1].split("```")[0].strip()
    
    try:
        return json.loads(raw_content)
    except Exception:
        try:
            return json5.loads(raw_content)
        except Exception as e:
            raise ValueError(
                f"Failed to parse model JSON: {e}\nRaw content:\n{raw_content}"
            )
