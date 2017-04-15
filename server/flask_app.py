from flask import Flask, request, jsonify

import numpy as np

from server.exceptions import InvalidUsage
from server.envs import Envs
from server.utils import get_required_param, get_optional_param

app = Flask(__name__)
envs = Envs()


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


########## API route definitions ##########
@app.route('/v1/envs/', methods=['POST'])
def env_create():
    """
    Create an instance of the specified environment

    Parameters:
        - env_id: gym environment ID string, such as 'CartPole-v0'
    Returns:
        - instance_id: a short identifier (such as '3c657dbc')
        for the created environment instance. The instance_id is
        used in future API calls to identify the environment to be
        manipulated
    """
    env_id = get_required_param(request.get_json(), 'env_id')
    instance_id = envs.create(env_id)
    return jsonify(instance_id = instance_id)


@app.route('/v1/envs/', methods=['GET'])
def env_list_all():
    """
    List all environments running on the server

    Returns:
        - envs: dict mapping instance_id to env_id
        (e.g. {'3c657dbc': 'CartPole-v0'}) for every env
        on the server
    """
    all_envs = envs.list_all()
    return jsonify(all_envs = all_envs)


@app.route('/v1/envs/<instance_id>/reset/', methods=['POST'])
def env_reset(instance_id):
    """
    Reset the state of the environment and return an initial
    observation.

    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
        for the environment instance
    Returns:
        - observation: the initial observation of the space
    """
    observation = envs.reset(instance_id)
    if np.isscalar(observation):
        observation = observation.item()
    return jsonify(observation = observation)


@app.route('/v1/envs/<instance_id>/step/', methods=['POST'])
def env_step(instance_id):
    """
    Run one timestep of the environment's dynamics.

    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
        for the environment instance
        - action: an action to take in the environment
    Returns:
        - observation: agent's observation of the current
        environment
        - reward: amount of reward returned after previous action
        - done: whether the episode has ended
        - info: a dict containing auxiliary diagnostic information
    """
    json = request.get_json()
    action = get_required_param(json, 'action')
    render = get_optional_param(json, 'render', False)
    [obs_jsonable, reward, done, info] = envs.step(instance_id, action, render)
    return jsonify(observation = obs_jsonable,
                    reward = reward, done = done, info = info)


@app.route('/v1/envs/<instance_id>/action_space/', methods=['GET'])
def env_action_space_info(instance_id):
    """
    Get information (name and dimensions/bounds) of the env's
    action_space

    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
        for the environment instance
    Returns:
    - info: a dict containing 'name' (such as 'Discrete'), and
    additional dimensional info (such as 'n') which varies from
    space to space
    """
    info = envs.get_action_space_info(instance_id)
    return jsonify(info = info)


@app.route('/v1/envs/<instance_id>/action_space/sample', methods=['GET'])
def env_action_space_sample(instance_id):
    """
    Get a sample from the env's action_space

    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
        for the environment instance
    Returns:

    	- action: a randomly sampled element belonging to the action_space
    """  
    action = envs.get_action_space_sample(instance_id)
    return jsonify(action = action)


@app.route('/v1/envs/<instance_id>/action_space/contains/<x>', methods=['GET'])
def env_action_space_contains(instance_id, x):
    """
    Assess that value is a member of the env's action_space
    
    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
        for the environment instance
	- x: the value to be checked as member
    Returns:
        - member: whether the value passed as parameter belongs to the action_space
    """  

    member = envs.get_action_space_contains(instance_id, x)
    return jsonify(member = member)


@app.route('/v1/envs/<instance_id>/observation_space/', methods=['GET'])
def env_observation_space_info(instance_id):
    """
    Get information (name and dimensions/bounds) of the env's
    observation_space

    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
        for the environment instance
    Returns:
        - info: a dict containing 'name' (such as 'Discrete'),
        and additional dimensional info (such as 'n') which
        varies from space to space
    """
    info = envs.get_observation_space_info(instance_id)
    return jsonify(info = info)


@app.route('/v1/envs/<instance_id>/monitor/start/', methods=['POST'])
def env_monitor_start(instance_id):
    """
    Start monitoring.

    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
        for the environment instance
        - force (default=False): Clear out existing training
        data from this directory (by deleting every file
        prefixed with "openaigym.")
        - resume (default=False): Retain the training data
        already in this directory, which will be merged with
        our new data
    """
    j = request.get_json()

    directory = get_required_param(j, 'directory')
    force = get_optional_param(j, 'force', False)
    resume = get_optional_param(j, 'resume', False)
    video_callable = get_optional_param(j, 'video_callable', False)
    envs.monitor_start(instance_id, directory, force, resume, video_callable)
    return ('', 204)


@app.route('/v1/envs/<instance_id>/monitor/close/', methods=['POST'])
def env_monitor_close(instance_id):
    """
    Flush all monitor data to disk.

    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
          for the environment instance
    """
    envs.monitor_close(instance_id)
    return ('', 204)


@app.route('/v1/envs/<instance_id>/close/', methods=['POST'])
def env_close(instance_id):
    """
    Manually close an environment

    Parameters:
        - instance_id: a short identifier (such as '3c657dbc')
          for the environment instance
    """
    envs.env_close(instance_id)
    return ('', 204)


@app.route('/v1/upload/', methods=['POST'])
def upload():
    """
    Upload the results of training (as automatically recorded by
    your env's monitor) to OpenAI Gym.

    Parameters:
        - training_dir: A directory containing the results of a
        training run.
        - api_key: Your OpenAI API key
        - algorithm_id (default=None): An arbitrary string
        indicating the paricular version of the algorithm
        (including choices of parameters) you are running.
        """
    j = request.get_json()
    training_dir = get_required_param(j, 'training_dir')
    api_key      = get_required_param(j, 'api_key')
    algorithm_id = get_optional_param(j, 'algorithm_id', None)

    try:
        gym.upload(training_dir, algorithm_id, writeup=None, api_key=api_key,
                   ignore_open_monitors=False)
        return ('', 204)
    except gym.error.AuthenticationError:
        raise InvalidUsage('You must provide an OpenAI Gym API key')


@app.route('/v1/shutdown/', methods=['POST'])
def shutdown():
    """ Request a server shutdown - currently used by the integration tests to repeatedly create and destroy fresh copies of the server running in a separate thread"""
    f = request.environ.get('werkzeug.server.shutdown')
    f()
    return 'Server shutting down'