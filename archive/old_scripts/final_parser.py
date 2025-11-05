import re
from input import meeting

def parse_correctly(text):
    """
    Parses multi-line messages from the meeting transcript.
    A message starts after a speaker line and ends before the next speaker line or date.
    """
    messages = []
    current_date = None
    current_message_lines = []
    current_speaker_info = None

    # Regex to identify a line with a speaker and timestamp
    speaker_pattern = re.compile(r"^(\d{1,2}:\d{2}\s*(?:AM|PM))\s*—\s*([^:]+):")
    date_pattern = re.compile(r"^(Monday|Tuesday|Wednesday|Thursday|Friday), (\w+ \d{1,2}, \d{4})")

    for line in text.split('\n'):
        date_match = date_pattern.match(line)
        speaker_match = speaker_pattern.match(line)

        # If we find a new date or a new speaker, the previous message has ended.
        if (date_match or speaker_match) and current_speaker_info and current_message_lines:
            # Join the collected lines to form the full message
            full_message = ' '.join(current_message_lines).strip()
            
            # Clean up the message from quotes and extra whitespace
            full_message = re.sub(r'^"|"$', '', full_message) # Remove surrounding quotes
            full_message = re.sub(r'\s+', ' ', full_message) # Normalize whitespace

            if full_message: # Only save if there is content
                messages.append({
                    "date": current_speaker_info['date'],
                    "speaker": current_speaker_info['speaker'],
                    "message": full_message
                })
            
            # Reset for the next message
            current_message_lines = []

        # --- Start of a new entry ---
        if date_match:
            current_date = date_match.group(2)
            current_speaker_info = None # A new day starts
        
        if speaker_match and current_date:
            time, speaker_name = speaker_match.groups()
            
            # Clean up speaker name
            if '(' in speaker_name:
                speaker_name = speaker_name.split('(')[0].strip()

            current_speaker_info = {'date': current_date, 'speaker': speaker_name}
        
        # If we are after a speaker but before the next one, it's part of the message
        elif current_speaker_info and not speaker_match and not date_match:
            current_message_lines.append(line.strip())

    # Add the very last message in the file
    if current_speaker_info and current_message_lines:
        full_message = ' '.join(current_message_lines).strip()
        full_message = re.sub(r'^"|"$', '', full_message)
        full_message = re.sub(r'\s+', ' ', full_message)
        if full_message:
            messages.append({
                "date": current_speaker_info['date'],
                "speaker": current_speaker_info['speaker'],
                "message": full_message
            })

    return messages

if __name__ == "__main__":
    parsed_messages = parse_correctly(meeting)
    if parsed_messages:
        print(f"Successfully parsed {len(parsed_messages)} messages.")
        for msg in parsed_messages:
            print(f"Date: {msg['date']}, Speaker: {msg['speaker']}, Message: {msg['message']}")
    else:
        print("Failed to parse any messages.")
