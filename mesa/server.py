import socket
import traceback
from logger import logging as log

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
        func_args = [arg.strip() for arg in message.split(',')]
        func, args = func_args[0], func_args[1:]
        response = getattr(model, func)(*args)
        response = f'success: {response}' if response else "success: process exited ok"
        return response
    except Exception as e:
        log_traceback(e)
        response = f'error: {e}'
        return response


def model_worker_server(port, model):
    server_socket = get_server_socket(port)
    client_socket, client_address = server_socket.accept()

    while True:
        # Handle client request and respond here  
        received = client_socket.recv(1024).decode()
        log.info(f"Server {port} received: {received}")
        if received == 'kill':
            log.info(f"Server {port} received kill notification")
            response = 'OK'
        else:
            response = handle_message(model, received)

        encoded = response.encode()
        client_socket.send(encoded)
        if received == 'kill':
            time.sleep(1)
            client_socket.close()
            break

    log.info(f"Server {port} shutting down")
