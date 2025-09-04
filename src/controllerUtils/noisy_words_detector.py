def word_overlap_with_noise(word, noise_segments, threshold=0.8):
    """
    Returns True if at least `threshold` fraction of the word falls within any loud noise segment.
    """
    w_start = float(word['start'])
    w_end = float(word['end'])
    w_duration = w_end - w_start

    for noise in noise_segments:
        n_start = float(noise['start_time'])
        n_end = float(noise['end_time'])


        # Calculate overlap
        overlap_start = max(w_start, n_start)
        overlap_end = min(w_end, n_end)
        overlap_duration = max(0, overlap_end - overlap_start)

        if overlap_duration / w_duration >= threshold:
            return True
    return False


def get_words_in_loud_segments(words, noise_segments, threshold=0.8):
    return [
        {
            "word": word["word"],
            "start": float(word["start"]),
            "end": float(word["end"]),
            "score": float(word.get("score", 0.0))
        }
        for word in words
        if word_overlap_with_noise(word, noise_segments, threshold)
    ]