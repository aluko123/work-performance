from datetime import datetime

def parse_date(date_str: str) -> str | None:
    """Parses a date string from common formats into YYYY-MM-DD."""
    if not date_str or not isinstance(date_str, str):
        return None
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    print(f"Warning: Could not parse date string: {date_str}")
    return date_str

def parse_time(time_str: str) -> str | None:
    """Parses a time string from common formats into HH:MM:SS."""
    if not time_str or not isinstance(time_str, str):
        return None
    for fmt in ("%I:%M:%S %p", "%I:%M %p", "%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(time_str, fmt).strftime("%H:%M:%S")
        except ValueError:
            continue
    print(f"Warning: Could not parse time string: {time_str}")
    return time_str
