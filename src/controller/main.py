import sys
import os
# Add the parent directory to the Python path so we can import controllerUtils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllerUtils import (
    transcribe,
    detect_energy,
    get_words_in_loud_segments,
    word_overlap_with_noise,
    ctx_anomaly_detector,
    word_predictor
)

def print_low_confidence_words(whisper_res, thresh):
    print(f"Words below {thresh} confidence score:\n")
    for segment in whisper_res.get("segments", []):
        for word_info in segment.get("words", []):
            if word_info.get("score", 1.0) < thresh:
                print(f"'{word_info['word']}'  --  score: '{word_info['score']:.2f}'")
                
def print_loud_noise_segments(segments):
    print("Loud noise detection: \n")
    for segment in segments:
        if segment.get('label') == 'loud_noise':
            print(segment)

def segments_to_transcription(data):
    """
    Given WhisperX-style segmented output with word-level timing and scores,
    return the full raw transcription as a plain string.
    """
    words = []
    for segment in data.get('segments', []):
        for word_info in segment.get('words', []):
            words.append(word_info['word'])
    return ' '.join(words)

def indexed_transcription_str(text):

    """
    Returns a string where each word in the input text is followed by its index.
    Format: '0: word\n1: word\n...'
    """
    words = text.strip().split()
    lines = [f"{idx}: {word}" for idx, word in enumerate(words)]
    return "\n".join(lines)

def parse_indices_string(s):
    return [int(num.strip()) for num in s.split(',')]

def intersect_with_low_confidence_score(whisper_res, wrong_words_indices: list):
    """
    Given a WhisperX-style transcript and a list of word indexes,
    return a list of indexes where the word's confidence score < 0.3.
    """
    all_words = []
    for segment in whisper_res.get('segments', []):
        all_words.extend(segment.get('words', []))

    return [
        idx for idx in wrong_words_indices
        if 0 <= idx < len(all_words) and all_words[idx].get('score', 1.0) < 0.3
    ]

def adjust_by_noise_segments(all_words, loud_noise_segments):
    """
    Given a flat list of words (all_words) and a list of loud noise intervals,
    replaces words that overlap with '[blank]'. If no words overlap a noise interval,
    inserts a new '[blank]' word in the appropriate position in all_words.
    
    Modifies all_words in-place.
    """
    # Sort all_words by start time to enable ordered insertion
    all_words.sort(key=lambda w: float(w['start']))

    for noise in loud_noise_segments:
        if noise['label'] == 'loud_noise':
            start = float(noise['start_time'])
            end = float(noise['end_time'])

            matched = False
            for word in all_words:
                word_start = float(word['start'])
                word_end = float(word['end'])

                if word_start < end and word_end > start:
                    word['word'] = '[blank]'
                    word['start'] = max(word_start, start)
                    word['end'] = min(word_end, end)
                    matched = True

            if not matched:
                # Insert a new '[blank]' word into the correct position
                new_word = {'word': '[blank]', 'start': start, 'end': end, 'score': 0.0}
                inserted = False
                for i, word in enumerate(all_words):
                    if float(word['start']) > end:
                        all_words.insert(i, new_word)
                        inserted = True
                        break
                if not inserted:
                    all_words.append(new_word)

    # Ensure all_words remains sorted
    all_words.sort(key=lambda w: float(w['start']))
    return all_words


def insert_blank_and_modify_timestamp(whisper_res, indexes):
    """
    Adjust timestamps of words at given indexes:
    - start time is set to end time of previous word
    - end time is set to start time of next word

    Modifies the transcript in place.
    """
    # Flatten words into a list of (word_dict, segment_ref)
    all_words = []
    for segment in whisper_res.get('segments', []):
        for word in segment.get('words', []):
            all_words.append(word)

    for idx in indexes:
        if 1 <= idx < len(all_words) - 1:
            prev_word = all_words[idx - 1]
            curr_word = all_words[idx]
            next_word = all_words[idx + 1]

            curr_word['start'] = float(prev_word['end'])
            curr_word['word'] = '[blank]'
            curr_word['end'] = float(next_word['start'])
        # Optional: skip edge cases (first or last word) silently
    return all_words


def main():
    audio_path = "input/Audio test samples/HARVARD- CS LECTURE/masked/masked-cs-snippet.wav"
    
    
    whisper_result = transcribe(audio_path)
    whisper_transcription= segments_to_transcription(whisper_result)
    indexed_transcription = indexed_transcription_str(whisper_transcription)

    print("\n\n\n\n#### INITIAL WHISPER TRANSCRIPTION #### \n")
    print(whisper_transcription)

    #LLM call using a tailored system prompt for the task
    anomaly_detector_result = ctx_anomaly_detector(whisper_transcription, indexed_transcription)
    parsed_anomaly_detection_result = anomaly_detector_result.choices[0].message.content

    # detect high energy audio segments
    energy_segments = detect_energy(audio_path)

    print("\n\n\n #### TOOLS USED #### \n")

    print_loud_noise_segments(energy_segments)
    print("\n")
    print("Anomaly detection indexes: " ,parsed_anomaly_detection_result)
    print("\n")
    print_low_confidence_words(whisper_result, 0.3)

    
    #Insert placeholders to the transcription using the context anomally resulst and verify with loud segments time intervals
    indices_list = parse_indices_string(parsed_anomaly_detection_result)
    modified_transcription = insert_blank_and_modify_timestamp(whisper_result, indices_list)
    result = adjust_by_noise_segments(modified_transcription, energy_segments)   #
    blank_inserted_trans = ' '.join(word['word'] for word in result)

    print("\n\n\n #### FINAL TRANSCRIPTION WITH DETECTION PLACEHOLDERS #### \n")
    print(blank_inserted_trans)

    predicted_trans = word_predictor(blank_inserted_trans)
  

    print("\n\n\n\n #### FINAL RESULT ####\n")
    print(predicted_trans.choices[0].message.content)





if __name__ == "__main__":
    main()