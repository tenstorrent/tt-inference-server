import multiprocessing
import os
import queue
import random
import sys
import threading
import time
import uuid
import json
from threading import Lock
from typing import Optional

import jwt
from flask import Flask, Response, request, session, abort

sys.path.append(os.getcwd())

from llama3_70b_backend import run_backend
from inference_config import inference_config
from inference_logger import get_logger

logger = get_logger(__name__)
logger.info(f"importing {__name__}")
logger.info(json.dumps(inference_config._asdict(), indent=4))

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "test-app-secret")
INIT_ID = "COMPILE-INITIALIZATION"

HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_INTERNAL_SERVER_ERROR = 500
HTTP_SERVICE_UNAVAILABLE = 503


class Context:
    # Store current context
    # Store conversation history
    # Initialize the lock
    def __init__(self):
        self.conversations = {}
        self.user_status = {}  # {user_id:q_position}
        self.num_decoding_users = 0
        self.user_last_read = {}
        self.user_parameters = {}
        # Initialize the lock
        self.context_lock = Lock()

    def get_num_decoding_users(self):
        with self.context_lock:
            return self.num_decoding_users

    def set_num_decoding_users(self, value):
        with self.context_lock:
            self.num_decoding_users = value


# Shared variables with a lock for thread-safe access
context = Context()
time_last_response = time.time()
time_last_response_lock = Lock()
api_log_dir = os.path.join(inference_config.log_cache, "api_logs")


def initialize_decode_backend():
    global input_queue
    global output_queue
    global status_queue
    global output_queue_map
    global output_queue_map_lock

    output_queue_map = {}
    output_queue_map_lock = threading.Lock()

    input_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()
    status_queue = multiprocessing.Queue()
    # run the decode backend in a separate process
    backend_process = multiprocessing.Process(
        target=run_backend,
        args=(
            input_queue,
            output_queue,
            status_queue,
            inference_config.backend_debug_mode,
        ),
    )
    backend_process.start()
    default_params, _ = get_user_parameters({"max_tokens": 4})
    input_queue.put((INIT_ID, "Dummy input for initialization", default_params))
    respond_to_users_thread = threading.Thread(target=respond_to_users)
    respond_to_users_thread.start()
    status_func_thread = threading.Thread(target=status_func)
    status_func_thread.start()


def _garbage_collection():
    """reclaim resources for output queues for user_ids that are:
    1. not in self.users (have completed generation)
    2. are empty (have been read out by request handling thread)

    Only this function deletes from the output_queue_map in a single thread.
    """
    current_time = time.time()

    with context.context_lock:
        active_user_ids = {
            user_id
            for user_id, last_read_time in context.user_last_read.items()
            if current_time - last_read_time < inference_config.max_inactive_seconds
        }
    marked_for_deletion = set()
    with output_queue_map_lock:
        for user_id, output_q in output_queue_map.items():
            if user_id not in active_user_ids and output_q.empty():
                marked_for_deletion.add(user_id)

    for user_id in marked_for_deletion:
        with output_queue_map_lock:
            del output_queue_map[user_id]

        if user_id in context.user_last_read.keys():
            with context.context_lock:
                del context.user_last_read[user_id]


def _update_time_last_response():
    # only respond_to_users thread should update this value
    global time_last_response
    with time_last_response_lock:
        time_last_response = time.time()


def get_time_last_response():
    with time_last_response_lock:
        return time_last_response


def respond_to_users():
    MAX_USER_ROWS = 32
    while True:
        # q.get() will block the thread until output received
        response_session_id, response = output_queue.get()
        _update_time_last_response()
        if response_session_id == INIT_ID:
            continue
        with output_queue_map_lock:
            if response_session_id not in output_queue_map:
                output_queue_map[response_session_id] = queue.Queue()
            output_queue_map[response_session_id].put(response)
        if inference_config.frontend_debug_mode:
            # Log response
            with open(f"{api_log_dir}/response_{response_session_id}.txt", "a") as f:
                f.write(response)


def status_func():
    global context
    time_last_keep_alive_input = time.time()
    time_last_status_msg = time.time()
    NON_RESPONSE_TIME_FOR_HANG = inference_config.keepalive_input_period_seconds * 5
    while True:
        time.sleep(1.0)
        # read status queue from backend
        if not status_queue.empty():
            (
                prompt_q_size,
                num_decoding_users,
                decoding_users,
            ) = status_queue.get_nowait()
            logger.info(f"num_decoding_users: {num_decoding_users}")
            logger.info(f"prompt_q_size: {prompt_q_size}")
            context.set_num_decoding_users(num_decoding_users)
            time_last_status_msg = time.time()
        # update vars
        time_since_response = time.time() - get_time_last_response()
        time_since_keep_live = time.time() - time_last_keep_alive_input
        time_since_status_msg = time.time() - time_last_status_msg
        # send keep alive prompt
        if (
            time_since_response > inference_config.keepalive_input_period_seconds
            and time_since_keep_live > inference_config.keepalive_input_period_seconds
        ):
            session_id = "KEEP-ALIVE-INPUT"
            prompt = "the"
            params, _ = get_user_parameters(data={"max_tokens": 2})
            input_queue.put((session_id, prompt, params))
            time_last_keep_alive_input = time.time()
            logger.info(
                f"keep alive: time_since_response={time_since_response}, time_since_keep_live={time_since_keep_live}"
            )
        # check status
        if time_since_response > NON_RESPONSE_TIME_FOR_HANG:
            logger.error(
                f"Model backend is hanging. time_since_response:={time_since_response}, time_since_status_msg:={time_since_status_msg}"
            )
        # Note: only this thread should perform garbage collection to avoid lock contention
        _garbage_collection()


def preprocess_prompt(data):
    prompt, error = safe_convert_type(
        data_dict=data, key="text", dest_type=str, default=""
    )
    return prompt, error


def safe_convert_type(data_dict, key, dest_type, default):
    error = None
    converted_value = default
    if key in data_dict:
        value = data_dict.get(key, default)
        try:
            converted_value = dest_type(value)
        # pylint: disable=broad-except
        except Exception as err:
            logger.error(f"Error: safe_convert excepts: {err}")
            status_phrase = (
                f"Parameter: {key} is type={type(value)}, expected {dest_type}"
            )
            status_code = 400
            error = ({"message": status_phrase}, status_code)

    return converted_value, error


def apply_parameter_bounds(params):
    # clip parameters to within min / max boundaries
    error = None
    # (lower_bound, upper_bound)
    param_bounds = {
        "temperature": (0.01, 100.0),
        "top_p": (0.01, 1.0),
        "top_k": (1, 1000),
        "max_tokens": (1, 2048),
    }

    for key, (lower_bound, upper_bound) in param_bounds.items():
        value = params[key]
        within_bounds = lower_bound <= value <= upper_bound
        if not within_bounds:
            status_phrase = f"Parameter: {key}={value} is outside bounds, {lower_bound} <= {key} <= {upper_bound}."
            status_code = 400
            error = ({"message": status_phrase}, status_code)
            return {}, error
    return params, error


def get_user_parameters(data):
    """This function turns user input into parameters."""
    # (default_value, python_type)
    default_params = {
        "temperature": (1.0, float),
        "top_p": (0.9, float),
        "top_k": (10, int),
        "max_tokens": (1024, int),
        "stop_sequence": ("", str),
        "return_prompt": (False, bool),
    }
    error = None
    params = {}
    # user input sanitization to expected python types, or default values, with error handling
    for key, (default_value, python_type) in default_params.items():
        value, error = safe_convert_type(
            data_dict=data, key=key, dest_type=python_type, default=default_value
        )
        if error is not None:
            # return 400 to user on first error
            return {}, error
        params[key] = value

    return params, error


def sanitize_request(request):
    error = None
    user_session_id = None

    if request.is_json:
        data = request.get_json()
    else:
        error = {"message": "Request was not JSON"}, 400
        return None, None, None, error

    prompt, error = preprocess_prompt(data)
    if error:
        return None, None, None, error

    params, error = get_user_parameters(data)
    if error:
        return None, None, None, error

    if not prompt:
        error = {
            "message": "required 'text' parameter is either empty or not provided"
        }, 400
        return None, None, None, error

    params, error = apply_parameter_bounds(params)
    if error:
        return None, None, None, error

    if "session_id" in data:
        user_session_id, error = safe_convert_type(data, "session_id", str, None)
        if error:
            return None, None, None, error

    return prompt, params, user_session_id, error


def get_output(session_id):
    done_generation = False
    started_generation = False
    while not done_generation:
        if session_id in output_queue_map and not started_generation:
            started_generation = True
        elif session_id not in output_queue_map and not started_generation:
            # waiting for start of generation
            time.sleep(0.02)
            continue
        elif session_id not in output_queue_map and started_generation:
            # generation ended without EOS token
            logger.error(f"session_id: {session_id} ended without EOS.")
            done_generation = True
            continue

        # use nowait and continue sleep loop to avoid reading from q after slot_idx reallocated
        if output_queue_map[session_id].empty():
            time.sleep(0.02)
            continue

        out_text = output_queue_map[session_id].get_nowait()
        with context.context_lock:
            context.user_last_read[session_id] = time.time()
        if out_text.endswith(inference_config.end_of_sequence_str):
            done_generation = True

        if inference_config.frontend_debug_mode:
            with open(f"{api_log_dir}/user_{session_id}.txt", "a") as f:
                f.write(out_text)

        yield out_text


def handle_inference(prompt, params, user_session_id):
    global context
    error = None
    # create a session_id if not supplied
    if "session_id" not in session and user_session_id is None:
        session["session_id"] = str(uuid.uuid4())
    else:
        logger.info(
            f"user attemping pre-existing session: {session.get('session_id')}, {user_session_id}"
        )
        # currently only support stateless sessions
        session["session_id"] = str(uuid.uuid4())

    # if input_q full, retry with simple back-off
    for timeout in [0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]:
        if input_queue.qsize() >= inference_config.max_input_qsize:
            # add jitter
            sleep_t = timeout + 0.025 * random.random()
            time.sleep(sleep_t)
        else:
            break
    else:
        error = {"message": "Service overloaded, try again later."}, 503
        return None, error

    # input
    session_id = session.get("session_id")
    input_queue.put((session_id, prompt, params))

    if inference_config.frontend_debug_mode:
        # Log user's prompt
        with open(f"{api_log_dir}/prompt_{session_id}.txt", "a") as f:
            f.write("Prompt:\n" + prompt + "\n")

    return session_id, error


def chat_prompt_preprocessing(prompt):
    preprocessed_prompt = f"User: {prompt}\nAI: "
    return preprocessed_prompt


def get_chat_output(session_id):
    # the chat interface expects the fullmessage in each call
    # and the event / data syntax in the response string
    full_message = ""
    for chunk in get_output(session_id):
        # backend handles spacing after words
        if chunk != "<|endoftext|>":
            full_message += chunk
            yield f"event: answer\ndata: {json.dumps({'message': full_message})}\n\n"
    # after <|endoftext|> no more messages
    yield 'event: close\ndata: {"message": "Connection closed"}\n\n'


def normalize_token(token) -> [str, str]:
    """
    Note that scheme is case insensitive for the authorization header.
    See: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Authorization#directives
    """  # noqa: E501
    one_space = " "
    words = token.split(one_space)
    scheme = words[0].lower()
    return [scheme, " ".join(words[1:])]


def read_authorization(
    headers,
) -> Optional[dict]:
    authorization = headers.get("authorization")
    if not authorization:
        abort(HTTP_UNAUTHORIZED, description="Must provide Authorization header.")
    [scheme, parameters] = normalize_token(authorization)
    if scheme != "bearer":
        user_error_msg = f"Authorization scheme was '{scheme}' instead of bearer"
        abort(HTTP_UNAUTHORIZED, description=user_error_msg)
    try:
        payload = jwt.decode(parameters, os.getenv("JWT_SECRET"), algorithms=["HS256"])
        if not payload:
            abort(HTTP_UNAUTHORIZED)
        return payload
    except jwt.InvalidTokenError as exc:
        user_error_msg = f"JWT payload decode error: {exc}"
        abort(HTTP_BAD_REQUEST, description=user_error_msg)


@app.route(f"/chat/{inference_config.inference_route_name}", methods=["POST"])
def chat_inference_formatted():
    _ = read_authorization(request.headers)
    # user will get 400 on invalid input, with helpful status message
    prompt, params, user_session_id, error = sanitize_request(request)
    if error:
        return error
    preprocessed_prompt = chat_prompt_preprocessing(prompt)
    session_id, error = handle_inference(preprocessed_prompt, params, user_session_id)
    if error:
        return error

    # output
    return Response(get_chat_output(session_id), content_type="text/event-stream")


@app.route(f"/inference/{inference_config.inference_route_name}", methods=["POST"])
def inference():
    _ = read_authorization(request.headers)
    # user will get 400 on invalid input, with helpful status message
    prompt, params, user_session_id, error = sanitize_request(request)
    if error:
        return error
    session_id, error = handle_inference(prompt, params, user_session_id)
    if error:
        return error
    # output
    return Response(get_output(session_id), content_type="text/event-stream")


@app.route("/")
@app.route("/health")
def health_check():
    # check for keep alive failures occuring for 5x periods
    time_since_response = time.time() - get_time_last_response()
    if time_since_response > (inference_config.keepalive_input_period_seconds * 5):
        return (
            "Keep alive prompts failed five times. Service unhealthy.",
            HTTP_INTERNAL_SERVER_ERROR,
        )

    return "OK", 200


backend_initialized = False


def global_backend_init():
    global backend_initialized
    if not backend_initialized:
        # Create server log directory
        if not os.path.exists(api_log_dir):
            os.makedirs(api_log_dir)
        initialize_decode_backend()
        backend_initialized = True


def create_server():
    logger.info("Starting inference API server ...")
    global_backend_init()
    return app


# NOTE: this server should be run using gunicorn instead of app.run()
# To run the server for local development and testing use the mock server:
# _mock_inference_api_server.py
