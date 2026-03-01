def format_duration(seconds: float) -> str:
    """Format a duration in seconds into a string of the format HH:MM:SS.mmm
    where HH is hours, MM is minutes, SS is seconds, and mmm is milliseconds.
    """
    ms = int((seconds - int(seconds)) * 1000)
    secs_total = int(seconds)
    hours, rem = divmod(secs_total, 3600)
    minutes, seconds_ = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_:02d}.{ms:03d}"