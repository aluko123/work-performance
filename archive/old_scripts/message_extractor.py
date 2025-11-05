import re
from input import meeting

def extract_messages(text):
    message_pattern = re.compile(r"^(\d{1,2}:\d{2}\s*(?:AM|PM))\s*—\s*([^:]+):\s*(.*)", re.MULTILINE)
    date_pattern = re.compile(r"^(Monday|Tuesday|Wednesday|Thursday|Friday), (\w+ \d{1,2}, \d{4})")

    messages = []
    current_date = None

    for line in text.split('\n'):
        date_match = date_pattern.match(line)
        if date_match:
            current_date = date_match.group(2)
            continue

        message_match = message_pattern.match(line)
        if message_match and current_date:
            time, speaker, content = message_match.groups()
            speaker = speaker.strip()
            if '(' in speaker:
                speaker = speaker.split('(')[0].strip()
            
            messages.append({
                "date": current_date,
                "time": time,
                "speaker": speaker,
                "message": content.strip()
            })

    for msg in messages:
        print(f"Date: {msg['date']}, Speaker: {msg['speaker']}, Message: {msg['message']}")

if __name__ == "__main__":
    extract_messages(meeting)
