import numpy as np  
import itertools
from typing import List
def wer(truth: List[str],
        hypothesis: List[str],
        ) -> float:
    """
    Calculate the WER between a ground-truth string and a hypothesis string

    :param truth the ground-truth sentence as a string or list of words
    :param hypothesis the hypothesis sentence as a string or list of words
    :return: the WER, the distance (also known as the amount of
    substitutions, insertions and deletions) and the length of the ground truth
    """

    # Create the list of vocabulary used
    w2i = dict.fromkeys(truth + hypothesis)
    for i,w in enumerate(w2i):
        w2i[w] = i
    # recreate the truth and hypothesis string as a list of tokens
    t = []
    h = []

    for w in truth:
        t.append(w2i[w])

    for w in hypothesis:
        h.append(w2i[w])

    # now that the words are tokenized, we can do alignment
    distance = _edit_distance(t, h)

    # and the WER is simply distance divided by the length of the truth
    n = len(truth)
    if n == 0:
        return 0
    error_rate = distance / n

    return error_rate
          
def _edit_distance(a: List[int], b:List[int]) -> int:
    """
    Calculate the edit distance between two lists of integers according to the
    Wagner-Fisher algorithm. Reference:
    https://en.wikipedia.org/wiki/Wagner%E2%80%93Fischer_algorithm)

    :param a: the list of integers representing a string, where each integer is
    a single character or word
    :param b: the list of integers representing the string to compare distance
    with
    :return: the calculated distance
    """
    if len(a) == 0:
        return len(b)
    elif len(b) == 0:
        return len(a)

    # Initialize the matrix/table and set the first row and column equal to
    # 1, 2, 3, ...
    # Each column represent a single token in the reference string a
    # Each row represent a single token in the reference string b
    #
    m = np.zeros((len(b) + 1, len(a) + 1)).astype(dtype=np.int32)

    m[0, 1:] = np.arange(1, len(a) + 1)
    m[1:, 0] = np.arange(1, len(b) + 1)

    # Now loop over remaining cell (from the second row and column onwards)
    # The value of each selected cell is:
    #
    #   if token represented by row == token represented by column:
    #       value of the top-left diagonal cell
    #   else:
    #       calculate 3 values:
    #            * top-left diagonal cell + 1 (which represents substitution)
    #            * left cell + 1 (representing deleting)
    #            * top cell + 1 (representing insertion)
    #       value of the smallest of the three
    #
    for i in range(1, m.shape[0]):
        for j in range(1, m.shape[1]):
            if a[j-1] == b[i-1]:
                m[i, j] = m[i-1, j-1]
            else:
                m[i, j] = min(
                    m[i-1, j-1] + 1,
                    m[i, j - 1] + 1,
                    m[i - 1, j] + 1
                )

    # and the minimum-edit distance is simply the value of the down-right most
    # cell

    return m[len(b), len(a)]