# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
TT-Comfy Bridge Server - Main entry point.

Unix domain socket server that exposes tt-metal models to ComfyUI.
"""

import asyncio
import argparse
import logging
import signal
import sys
from pathlib import Path

from server.config import BridgeConfig
from server.model_registry import ModelRegistry
from server.operations import OperationHandler
from protocol.serialization import MessageSerializer
from protocol.messages import Request, Response


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TTComfyBridgeServer:
    """
    Unix socket server for TT-Comfy Bridge.
    
    Handles IPC communication with ComfyUI clients and routes operations
    to tt-metal model runners.
    """
    
    def __init__(self, config: BridgeConfig):
        self.config = config
        self.registry = ModelRegistry()
        self.handler = OperationHandler(self.registry)
        self.server = None
        self.running = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.running = False
        if self.server:
            self.server.close()
    
    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """
        Handle a client connection.
        
        Args:
            reader: Stream reader for receiving messages
            writer: Stream writer for sending responses
        """
        addr = writer.get_extra_info('peername', 'unknown')
        logger.info(f"Client connected: {addr}")
        
        try:
            while self.running:
                # Read request message
                try:
                    message_data = await MessageSerializer.read_message(reader)
                except asyncio.IncompleteReadError:
                    logger.info(f"Client {addr} disconnected")
                    break
                except Exception as e:
                    logger.error(f"Error reading message from {addr}: {e}")
                    break
                
                # Parse request
                try:
                    request = Request.from_dict(message_data)
                    logger.debug(f"Received request: {request.operation} (id={request.request_id})")
                except Exception as e:
                    logger.error(f"Failed to parse request: {e}")
                    response = Response.error(f"Invalid request format: {e}")
                    await MessageSerializer.write_message(writer, response.to_dict())
                    continue
                
                # Handle operation
                response = await self.handler.handle_operation(
                    operation=request.operation,
                    data=request.data,
                    request_id=request.request_id
                )
                
                # Send response
                try:
                    await MessageSerializer.write_message(writer, response.to_dict())
                    logger.debug(f"Sent response: {response.status} (id={response.request_id})")
                except Exception as e:
                    logger.error(f"Error sending response to {addr}: {e}")
                    break
                
        except Exception as e:
            logger.error(f"Error handling client {addr}: {e}", exc_info=True)
        finally:
            writer.close()
            await writer.wait_closed()
            logger.info(f"Client {addr} connection closed")
    
    async def start(self):
        """Start the bridge server."""
        # Ensure socket directory exists
        self.config.socket_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove existing socket if present
        if self.config.socket_path.exists():
            logger.info(f"Removing existing socket: {self.config.socket_path}")
            self.config.socket_path.unlink()
        
        # Start Unix socket server
        logger.info(f"Starting TT-Comfy Bridge Server on {self.config.socket_path}")
        self.server = await asyncio.start_unix_server(
            self.handle_client,
            path=str(self.config.socket_path)
        )
        
        self.running = True
        logger.info(f"Server ready, listening on {self.config.socket_path}")
        
        # Serve forever
        async with self.server:
            await self.server.serve_forever()
    
    async def shutdown(self):
        """Shutdown the server gracefully."""
        logger.info("Shutting down server...")
        self.running = False
        
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        # Cleanup resources
        self.handler.tensor_bridge.cleanup_all()
        self.registry.clear()
        
        # Remove socket file
        if self.config.socket_path.exists():
            self.config.socket_path.unlink()
        
        logger.info("Server shutdown complete")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TT-Comfy Bridge Server - Unix socket server for ComfyUI integration"
    )
    parser.add_argument(
        "--socket-path",
        type=str,
        default="/tmp/tt-comfy.sock",
        help="Path to Unix domain socket (default: /tmp/tt-comfy.sock)"
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Tenstorrent device ID (default: 0)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create configuration
    config = BridgeConfig(
        socket_path=args.socket_path,
        device_id=args.device_id,
        log_level=args.log_level
    )
    
    # Create and start server
    server = TTComfyBridgeServer(config)
    
    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        return 1
    finally:
        await server.shutdown()
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

