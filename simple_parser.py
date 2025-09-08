from input import meeting

def simple_parse(text):
    lines = text.split('\n')
    messages = []
    current_date = None
    for line in lines:
        if not line.strip():
            continue
        try:
            if line.startswith("Monday,") or line.startswith("Tuesday,") or line.startswith("Wednesday,") or line.startswith("Thursday,") or line.startswith("Friday,"):
                current_date = line.split(",")[1].strip()
            elif "AM —" in line or "PM —" in line:
                parts = line.split("—")
                speaker_part = parts[1].split(":")[0].strip()
                if '(' in speaker_part:
                    speaker = speaker_part.split('(')[0].strip()
                else:
                    speaker = speaker_part
                if len(parts[1].split(":")) > 1:
                    message_text = parts[1].split(":")[1].strip()
                else:
                    message_text = ""
                messages.append({
                    "date": current_date,
                    "speaker": speaker,
                    "message": message_text
                })
        except Exception as e:
            print(f"Error parsing line: {line}")
            print(e)
    
    for msg in messages:
        print(f"Date: {msg['date']}, Speaker: {msg['speaker']}, Message: {msg['message']}")

if __name__ == "__main__":
    simple_parse(meeting)