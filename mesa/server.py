import socket
import traceback
from logger import logging as log
import pickle
import time

def log_traceback(ex, ex_traceback=None):
    ex_traceback = ex.__traceback__
    tb_lines = [ line.rstrip('\n') for line in
                 traceback.format_exception(ex.__class__, ex, ex_traceback)]
    log.error(tb_lines)


def get_server_socket(port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', port))
    server_socket.listen(1)
    log.info(f"Server process listening on port {port}")
    return server_socket


def handle_message(model, message):
    try:
        if len(message) == 2:
             is_method = True
             args = message[1]
        elif len(message) == 1: # else, assume it's an attribute:
             is_method = False
        else: # Can't understand this message
            raise ValueError("Message with more than two parts passed")

        assert isinstance(message[0], str), f"Can't understand this attribute or method: {message[0]}"

        path = message[0].split('.')
        attr = model
        for subattr in path:
            attr = getattr(obj, subattr)
        
        response = attr(*args) if is_method else attr
        return response

    except Exception as e:
        log_traceback(e)
        response = e
        return response

def model_worker_server(port, model):
    server_socket = get_server_socket(port)
    client_socket, client_address = server_socket.accept()

    while True:
        # Handle client request and respond here  
        received = receive(client_socket)
        log.info(f"Server {port} received: {received}")

        if received == 'kill':
            log.info(f"Server {port} received kill notification")
            response = 'OK'
        else:
            response = handle_message(model, received)

        send(response, client_socket)

        if received == 'kill':
            time.sleep(1)
            client_socket.close()
            break

    log.info(f"Server on port {port} shutting down")


def send(data, socket_):
    content = pickle.dumps(data)
    header = len(content).to_bytes(8, 'little')
    response = header + content
    socket_.send(response)


def receive(socket_):
    response = b''
    # Wait for data 
    header = socket_.recv(8)
    # decode the response
    length = int.from_bytes(header, 'little')
    data = b''
    received = 0
    while received < length:
        to_get = min(1024, length - received)
        data += socket_.recv(to_get)
        received += to_get
    response = pickle.loads(data)
    return response


def communicate_message(worker, message):
    client_socket = worker['connection']
    send(message, client_socket)
    response = receive(client_socket)
    return response
