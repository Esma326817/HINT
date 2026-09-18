import asyncio
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def _process_request(self, connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        elif request.path.startswith("/switch"):
            from urllib.parse import urlparse, parse_qs
            import json
            
            parsed = urlparse(request.path)
            query = parse_qs(parsed.query)
            
            ckpt_dir = query.get("dir", [None])[0]
            config_name = query.get("config", [None])[0]
            default_prompt = query.get("default_prompt", [None])[0]
            
            if not ckpt_dir or not config_name:
                return connection.respond(http.HTTPStatus.BAD_REQUEST, "Missing dir or config param\n")
            
            logger.info(f"Switching policy to {config_name} at {ckpt_dir} with prompt {default_prompt}")
            
            try:
                # Local import to avoid circular dependencies if any
                from openpi.training import config as _config
                from openpi.policies import policy_config as _policy_config
                
                # Delete old policy explicitly to free memory
                if hasattr(self, "_policy") and self._policy is not None:
                    del self._policy
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Create the new policy
                # Wait for loading. This will block the asyncio event loop for a few seconds.
                train_config = _config.get_config(config_name)
                new_policy = _policy_config.create_trained_policy(
                    train_config, 
                    ckpt_dir, 
                    default_prompt=default_prompt
                )
                
                self._policy = new_policy
                self._metadata = new_policy.metadata
                logger.info("Policy switch completed.")
                return connection.respond(http.HTTPStatus.OK, json.dumps({"status": "success"}))
                
            except Exception as e:
                logger.error(f"Failed to switch policy: {e}")
                logger.error(traceback.format_exc())
                return connection.respond(http.HTTPStatus.INTERNAL_SERVER_ERROR, f"Error: {e}\n")

        # Continue with the normal websocket handling.
        return None

    # Replace the top level handle function in start 
    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=self._process_request,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                
                # Squeeze first dimension if it's 1 (remove batch dimension if present)
                def squeeze_first_dim(value):
                    if isinstance(value, dict):
                        return {k: squeeze_first_dim(v) for k, v in value.items()}
                    elif isinstance(value, (list, tuple)):
                        return type(value)(squeeze_first_dim(v) for v in value)
                    elif hasattr(value, 'shape') and len(value.shape) > 0 and value.shape[0] == 1:
                        return value.squeeze(0)
                    return value
                
                obs = squeeze_first_dim(obs)
                if "observation/state" in obs:
                    if obs["observation/state"].shape[0] == 1:
                        obs["observation/state"] = obs["observation/state"].squeeze(0)
   
                infer_time = time.monotonic()
                
                # IMPORTANT for listener: Handle cases where the policy is currently None
                if getattr(self, "_policy", None) is None:
                    # Model not loaded yet, send error back as a raw string so client raises RuntimeError
                    await websocket.send("Error: No model loaded. Please switch to a checkpoint first.")
                    continue

                action = self._policy.infer(obs)
                
                # Check if action actually has actions key, else log it 
                if not isinstance(action, dict) or "actions" not in action:
                    logger.warning(f"Policy infer returned dictionary without 'actions' key! Keys: {action.keys() if isinstance(action, dict) else type(action)}")
                
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception as e:
                logger.error(f"Error during infer handler: {e}")
                logger.error(traceback.format_exc())
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise
