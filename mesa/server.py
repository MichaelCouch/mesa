import traceback
import pickle
import time
#import cProfile
import os
import multiprocessing
import logging as log
import signal

def ignore_signal_handler(signal, frame):
    # Use this to allow the other mechanisms to allow graceful shutdown
    log.debug(f"Ignoring signal {signal}")


class WorkerException(Exception):

    """Docstring for WorkerException. """

    def __init__(self, *args, **kwargs):
        """TODO: to be defined. """
        Exception.__init__(self, *args, **kwargs)



def log_traceback(ex, ex_traceback=None):
    ex_traceback = ex.__traceback__
    tb_lines = [ line.rstrip('\n') for line in
                 traceback.format_exception(ex.__class__, ex, ex_traceback)]
    log.error(tb_lines)

def shutdown_worker(worker):
    """shutdown the worker."""
    response = communicate_message(worker, 'kill')
    # This part might be overkill - joining the process is probably sufficient
    assert response == 0, "Unexpected response to kill command detected"
    # Wait for the worker to close the connection
    try:
        worker['connection'].recv()
    except EOFError:
        # This is expected when the worker closes the pipe
        worker['connection'].close()
        worker['process'].join()
        return

    raise RuntimeError(f"Received unexpected data while killing worker process")


def initialize_worker(child_model, pickle_path=None):
    """Create a worker process and return the connection details
    :returns: connection details for the worker

    """
    parent_pipe, child_pipe = multiprocessing.Pipe()

    process = multiprocessing.Process(
        target=model_worker_server,
        args=(child_pipe, child_model),
        kwargs={'pickle_path': pickle_path}
    )
    process.start()
    worker = {'process': process, 'connection': parent_pipe}
    if communicate_message(worker, "Ready?"):
        return worker
    raise WorkerException("Child worker failed to initialize")

def handle_message(model, message):
    try:
        if len(message) == 2:
             is_method = True
             args = [model if arg == 'target_model_self' else arg for arg in message[1]]
        elif len(message) == 1: # else, assume it's an attribute:
             is_method = False
        else: # Can't understand this message
            raise ValueError("Message with more than two parts passed")

        assert isinstance(message[0], str), f"Can't understand this attribute or method: {message[0]}"

        path = message[0].split('.')
        attr = model
        for subattr in path:
            attr = getattr(attr, subattr)
        response = attr(*args) if is_method else attr
        return response

    except Exception as e:
        log_traceback(e)
        response = e
        return response

def model_worker_server(pipe, model, pickle_path=None):
    if pickle_path is not None:
        with open(pickle_path, 'rb') as f:
            model = pickle.load(f)

    signal.signal(signal.SIGINT, ignore_signal_handler)
    #profiler = cProfile.Profile()
    #profiler.enable()
    log.debug(f"Child model server {model} running")
    while True:
        # Receieve instructions and respond
        received = receive(pipe)
        log.debug(f"Child model server {model} received: {received}")

        if received == 'kill':
            log.debug(f"Child model server {model} kill notification")
            response = 0
        elif received == 'Ready?':
            response = True
        else:
            response = handle_message(model, received)

        send(response, pipe)

        #profiler.dump_stats(f"profiling/server_{os.getpid()}.prof")
        #profiler.enable()
        if received == 'kill':
            pipe.close()
            break

    if model.grid is not None:
        model.grid.__exit__(None, None, None)
    for attr_collection in model._shared_attributes.values():
        attr_collection.__exit__(None, None, None)
    log.debug(f"Child model server {model} shutting down")
    #profiler.disable()
    #profiler.dump_stats(f"profiling/server_{os.getpid()}.prof")


def send(data, pipe):
    pipe.send(data)


def receive(pipe):
    response = pipe.recv()
    if isinstance(response, Exception):
        log_traceback(response)
        raise response
    return response


def communicate_message(worker, message):
    pipe = worker['connection']
    send(message, pipe)
    response = receive(pipe)
    return response
