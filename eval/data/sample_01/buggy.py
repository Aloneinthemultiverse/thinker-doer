def running_max(nums):
    """Return a list where each element is the max of nums seen so far."""
    out = []
    m = 0  # BUG: should start from the first element, not 0 (fails on negatives)
    for n in nums:
        if n > m:
            m = n
        out.append(m)
    return out
