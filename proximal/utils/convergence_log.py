import timeit


class ConvergenceLog(object):
    """A log of the runtime for an operation.
    """

    def __init__(self):
        self.total_time = 0.0
        self.iter_time = []
        self.objective_val = []
        self.lastticstamp = None

    def record_timing(self, elapsed):
        """Updates the log with the new time.
        """
        self.total_time += elapsed
        self.iter_time.append(self.total_time)

    def record_objective(self, val):
        """Updates the log with the new time.
        """
        self.objective_val.append(val)

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
