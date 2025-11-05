
import re
from input import meeting

def parse_by_date(text):
    date_pattern = re.compile(r"^(Monday|Tuesday|Wednesday|Thursday|Friday), (\w+ \d{1,2}, \d{4})")
    sections = date_pattern.split(text)
    print(sections)
    # The first element is usually an empty string before the first match
    if sections[0] == '' or sections[0].isspace():
        sections = sections[1:]

    daily_messages = []
    for i in range(0, len(sections), 3):
        if i + 2 < len(sections):
            day_of_week = sections[i]
            date_str = sections[i+1]
            content = sections[i+2]
            
            message_pattern = re.compile(r"^(\d{1,2}:\d{2}\s*(?:AM|PM))\s*—\s*([^:]+):\s*(.*)", re.MULTILINE)
            messages = message_pattern.findall(content)
            
            for msg in messages:
                time, speaker, message_text = msg
                speaker = speaker.strip()
                if '(' in speaker:
                    speaker = speaker.split('(')[0].strip()
                
                daily_messages.append({
                    "date": date_str.strip(),
                    "speaker": speaker,
                    "message": message_text.strip()
                })

    for msg in daily_messages:
        print(f"Date: {msg['date']}, Speaker: {msg['speaker']}, Message: {msg['message']}")

if __name__ == "__main__":
    parse_by_date(meeting)
