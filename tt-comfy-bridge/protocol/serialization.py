# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Message serialization for TT-Comfy Bridge IPC.
"""

import struct
import msgpack
from typing import Dict, Any


class MessageSerializer:
    """Handles message serialization/deserialization for IPC."""
    
    @staticmethod
    def serialize(data: Dict[str, Any]) -> bytes:
        """
        Serialize a message to bytes.
        
        Format: [4 bytes length][msgpack data]
        
        Args:
            data: Dictionary to serialize
            
        Returns:
            Serialized bytes with length prefix
        """
        # Pack data with msgpack
        packed = msgpack.packb(data, use_bin_type=True)
        
        # Prepend 4-byte length (big-endian unsigned int)
        length = len(packed)
        length_bytes = struct.pack('>I', length)
        
        return length_bytes + packed
    
    @staticmethod
    def deserialize(data: bytes) -> Dict[str, Any]:
        """
        Deserialize a message from bytes.
        
        Args:
            data: Bytes to deserialize (should not include length prefix)
            
        Returns:
            Deserialized dictionary
        """
        return msgpack.unpackb(data, raw=False)
    
    @staticmethod
    async def read_message(reader) -> Dict[str, Any]:
        """
        Read a complete message from an async stream reader.
        
        Args:
            reader: asyncio.StreamReader
            
        Returns:
            Deserialized message dictionary
        """
        # Read 4-byte length prefix
        length_bytes = await reader.readexactly(4)
        length = struct.unpack('>I', length_bytes)[0]
        
        # Read message data
        data = await reader.readexactly(length)
        
        return MessageSerializer.deserialize(data)
    
    @staticmethod
    async def write_message(writer, data: Dict[str, Any]):
        """
        Write a message to an async stream writer.
        
        Args:
            writer: asyncio.StreamWriter
            data: Dictionary to serialize and write
        """
        message = MessageSerializer.serialize(data)
        writer.write(message)
        await writer.drain()

