## -*- coding: utf8 -*-
## Code inspired by https://github.com/juliohm/HUM/blob/master/pyhum/utils.py
import logging
logger = logging.getLogger(__name__)

def get_mpi_world():

    #global MPI
    try:

        import mpi4py
        mpi4py.rc.threads = False
        mpi4py.rc.recv_mprobe = False

        from mpi4py import MPI as _MPI

    except ImportError:
        logger.error("Unable to initialize MPI communicator. Please check that mpi4py is correctly installed.")
        raise ImportError("Unable to initialize MPI communicator. Please check that mpi4py is correctly installed.")

    return _MPI

class MPIPool(object):
    """
    MPI-based parallel processing pool

    Design pattern in which a master process distributes tasks to other
    processes (a.k.a. workers) within a MPI communicator.

    Parameters
    ----------
    comm: mpi4py communicator, optional
        MPI communicator for transmitting messages
        Default: MPI.COMM_WORLD

    master: int, optional
        Master process is one of 0, 1,..., comm.size-1
        Default: 0

    debug: bool, optional
        Whether to print debugging information or not
        Default: False

    References
    ----------
    PACHECO, P. S., 1996. Parallel Programming with MPI.
    """
    def __init__(self, mpi=None, comm=None, master=0, parallel_comms=False):

        # get MPI world
        global MPI
        if mpi is None:
            MPI = get_mpi_world()
        else:
            MPI=mpi

        if comm == None:
            comm = MPI.COMM_WORLD

        # MPI pool must have at least 2 processes
        assert comm.size > 1

        # Master process must be in range [0,comm.size)
        assert 0 <= master < comm.size

        self.comm           = comm
        self.master         = master
        self.rank           = self.comm.Get_rank()
        self._processes     = int(self.comm.size)
        self.parallel_comms = parallel_comms
        self.workers        = set(range(comm.size))
        self.workers.discard(self.master)

        if self.rank == 0:
            logger.debug("Setting master process ...")
        else:
            logger.debug("Setting worker-{} process ...".format(self.rank))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def is_master(self):
        """
        Returns true if on the master process, false otherwise.
        """
        return self.rank == self.master

    def wait(self):
        """
        Make the workers listen to the master.
        """

        if self.is_master():
            return

        worker = self.rank
        status = MPI.Status()

        while True:

            logger.debug("Worker {0} waiting for task".format(worker))
            task = self.comm.recv(source=self.master, tag=MPI.ANY_TAG, status=status)

            if task is None:
                logger.debug("Worker {0} told to quit work".format(worker))
                break

            func, arg = task
            logger.debug("Worker {0} got task ({1}) with tag {2}".format(worker, func, status.tag))

            result = func(arg)

            logger.debug("Worker {0} sending answer ({1}) with tag {2}".format(worker, type(result), status.tag))
            self.comm.send(result, self.master, status.tag)


    def map(self, func, iterable):
        """
        Evaluate a function at various points in parallel. Results are
        returned in the requested order (i.e. y[i] = f(x[i])).
        """
        assert self.is_master()

        workerset   = self.workers.copy()
        tasklist    = [(tid, (func, arg)) for tid, arg in enumerate(iterable)]
        resultlist  = [None] * len(tasklist)
        pending     = len(tasklist)

        while pending:
            if workerset and tasklist:

                worker = workerset.pop()
                taskid, task = tasklist.pop()

                logger.debug("Sent task {0} to worker {1} with tag {2}".format(task[1], worker, taskid))
                req = self.comm.isend(task, dest=worker, tag=taskid)
                if not self.parallel_comms:
                    req.wait()

            if tasklist:
                flag = self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                if not flag: continue
            else:
                self.comm.Probe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)

            status = MPI.Status()
            result = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            worker = status.source
            taskid = status.tag
            logger.debug("Master received from worker {0} with tag {1}".format(worker, taskid))

            workerset.add(worker)
            resultlist[taskid] = result
            pending -= 1

        return resultlist

    def close(self):
        """
        Tell all the workers to quit.
        """
        if not self.is_master():
            return

        for worker in self.workers:
            self.comm.send(None, worker, 0)
