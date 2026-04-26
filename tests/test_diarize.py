from speech2md.diarize import Turn, Word, group_words


def test_group_words_starts_new_segment_on_speaker_change():
    words = [
        Word(0.0, 0.4, "hello"),
        Word(0.5, 0.9, "there"),
        Word(1.1, 1.4, "friend"),
    ]
    turns = [
        Turn(0.0, 1.0, "SPEAKER_00"),
        Turn(1.0, 2.0, "SPEAKER_01"),
    ]

    segs = group_words(words, turns, max_gap=1.0)

    assert [(seg.speaker, seg.text) for seg in segs] == [
        ("SPEAKER_00", "hello there"),
        ("SPEAKER_01", "friend"),
    ]


def test_group_words_prefers_next_turn_at_shared_boundary():
    words = [
        Word(0.0, 0.6, "alpha"),
        Word(0.8, 1.2, "beta"),
        Word(1.25, 1.6, "gamma"),
    ]
    turns = [
        Turn(0.0, 1.0, "SPEAKER_00"),
        Turn(1.0, 2.0, "SPEAKER_01"),
    ]

    segs = group_words(words, turns, max_gap=1.0)

    assert [(seg.speaker, seg.text) for seg in segs] == [
        ("SPEAKER_00", "alpha"),
        ("SPEAKER_01", "beta gamma"),
    ]


def test_group_words_uses_max_overlap_when_turns_overlap():
    words = [
        Word(0.0, 0.5, "alpha"),
        Word(0.95, 1.2, "beta"),
        Word(1.25, 1.5, "gamma"),
    ]
    turns = [
        Turn(0.0, 1.1, "SPEAKER_00"),
        Turn(1.0, 2.0, "SPEAKER_01"),
    ]

    segs = group_words(words, turns, max_gap=1.0)

    assert [(seg.speaker, seg.text) for seg in segs] == [
        ("SPEAKER_00", "alpha"),
        ("SPEAKER_01", "beta gamma"),
    ]



def test_group_words_can_shift_one_boundary_word_to_larger_pause():
    words = [
        Word(0.0, 0.3, "это"),
        Word(0.35, 0.7, "дадада"),
        Word(1.0, 1.12, "он"),
        Word(1.13, 1.5, "говорит"),
    ]
    turns = [
        Turn(0.0, 1.25, "SPEAKER_01"),
        Turn(1.25, 2.0, "SPEAKER_00"),
    ]

    segs = group_words(words, turns, max_gap=1.0)

    assert [(seg.speaker, seg.text) for seg in segs] == [
        ("SPEAKER_01", "это дадада"),
        ("SPEAKER_00", "он говорит"),
    ]
