"""
Utilities for evalutating probability density functions
"""


class unifsum(object):
    """
    pdf for sum of two uniform variables

    Args:
     a: lower limits (1st, 2nd)
     a: upper limits (1st, 2nd)
    """

    def __init__(self, a, b):
        #a1 = numpy.min(a)
        #b1 = numpy.max(a)
        if (a[0] < a[1]):
            a1 = a[0]
            b1 = a[1]
        else:
            a1 = a[1]
            b1 = a[0]

        #a2 = numpy.min(b)
        #b2 = numpy.max(b)
        if (b[0] < b[1]):
            a2 = b[0]
            b2 = b[1]
        else:
            a2 = b[1]
            b2 = b[0]

        w1 = b1 - a1
        w2 = b2 - a2
        #w = numpy.mean([w1, w2])
        w = (w1 + w2) / 2.0
        #self.c = numpy.mean([a2, b2]) + numpy.mean([a1, b1])
        self.c = (a1 + a2 + b1 + b2) / 2.0
        #self.w_min = numpy.min([w1, w2])
        self.w_min = w1 if w1 < w2 else w2

        #self.w_diff = numpy.abs(numpy.diff([w1, w2]))
        #self.w_diff = numpy.abs(w1-w2)
        self.w_diff = w1 - w2 if w1 > w2 else w2 - w1
        self.l = self.c - w
        self.h = self.c + w
        self.t = 1.0 / (self.w_min + self.w_diff)

    def __call__(self, p):
        """
        Evaluate density a point p

        Args:
         - p (float): point at which to evaluate the pdf

        Returns:
         (float): the pdf value
        """
        if (p < self.l):
            v = 0.0
        elif (p < (self.c - self.w_diff / 2)):
            v = self.t * (p - self.l) / self.w_min
        elif (p < (self.c + self.w_diff / 2)):
            v = self.t
        elif (p < self.h):
            v = self.t * (self.c + self.w_diff / 2 - p + self.w_min) / self.w_min
        else:
            v = 0.0
        return v
