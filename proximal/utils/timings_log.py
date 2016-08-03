import timeit


class TimingsEntry(object):
    """A log of the runtime for an operation.
    """

    def __init__(self, op):
        self.op = op
        self.evals = 0
        self.total_time = 0
        self.lastticstamp = None

    @property
    def avg_time(self):
        if self.evals == 0:
            return 0
        else:
            return self.total_time / self.evals

    def record_timing(self, elapsed):
        """Updates the log with the new time.
        """
        self.evals += 1
        self.total_time += elapsed

    def tic(self):
        """ Default timer
        Example: t = tic()
             ... code
             elapsed = toc(t)
             print( '{0}: {1:.4f}ms'.format(message, elapsed) )
        """

        t = timeit.default_timer()
        self.lastticstamp = t
        return t

    def toc(self):
        """ See tic f
        """

        # Last tic
        if self.lastticstamp is None:
            raise Exception('Error: Call to toc did never call tic before.')
        else:
            t = self.lastticstamp
            # Measure time in ms
            elapsed = (timeit.default_timer() - t) * 1000.0  # in ms
            # Update recrod.
            self.record_timing(elapsed)
            self.lastticstamp = None
            return elapsed

    def __str__(self):
        return "op = %s, evals = %s, total_time (ms) = %s, avg_time (ms) = %s" % (
            self.op, self.evals, self.total_time, self.avg_time)


class TimingsLog(object):
    """A log of the runtime for a set of operations.
    """

    def __init__(self, ops):
        self.ops = ops
        self.data = {}
        for op in self.ops:
            self.data[op] = TimingsEntry(op)

    def __getitem__(self, item):
        return self.data[item]

    def __str__(self):
        logs = []
        for op in self.ops:
            if self[op].evals > 0:
                logs += [str(self.data[op])]
        return '\n'.join(logs)
