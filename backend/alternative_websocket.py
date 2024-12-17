from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import logging
from websockets.client import connect
from typing import Dict

# Set up logging with more detailed formatting
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add file handler to also log to file
file_handler = logging.FileHandler('server.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
))
logger.addHandler(file_handler)

# Gemini Configuration
GEMINI_CONFIG = {
    "api_key": "",
    "model": "gemini-2.0-flash-exp",
    "base_uri": "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent"
}

GEMINI_SETUP_CONFIG = {
    "setup": {
        "model": f"models/{GEMINI_CONFIG['model']}",
        "generation_config": {
            "response_modalities": ["AUDIO"],
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {
                        "voice_name": "Puck"
                    }
                }
            }
        },
        "system_instruction": {
            "parts": [
                {
                    "text": "You are a friendly Gemini 2.0 model. Respond verbally in a casual, helpful tone."
                }
            ]
        }
    }
}

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GeminiConnection:
    def __init__(self):
        self.api_key = GEMINI_CONFIG["api_key"]
        self.model = GEMINI_CONFIG["model"]
        self.uri = f"{GEMINI_CONFIG['base_uri']}?key={self.api_key}"
        self.ws = None
        logger.info(f"Initialized GeminiConnection with model: {self.model}")

    async def connect(self):
        """Initialize connection to Gemini"""
        logger.info("Attempting to connect to Gemini websocket")
        try:
            self.ws = await connect(self.uri, extra_headers={"Content-Type": "application/json"})
            logger.info("Successfully connected to Gemini websocket")
        except Exception as e:
            logger.error(f"Failed to connect to Gemini websocket: {str(e)}", exc_info=True)
            raise
        
        logger.info("Sending setup message to Gemini")
        await self.ws.send(json.dumps(GEMINI_SETUP_CONFIG))
        
        # Wait for setup completion
        setup_response = await self.ws.recv()
        logger.info("Setup complete - received response from Gemini")
        # logger.debug(f"Setup response details: {setup_response}")
        return setup_response

    async def send_text(self, text: str):
        """Send text input to Gemini"""
        # logger.info("Preparing to send text to Gemini")
        msg = {
            "client_content": {
                "turn_complete": True,
                "turns": [{"role": "user", "parts": [{"text": text}]}],
            }
        }
        try:
            await self.ws.send(json.dumps(msg))
            logger.info("Successfully sent text to Gemini")
        except Exception as e:
            logger.error(f"Failed to send text: {str(e)}", exc_info=True)
            raise

    async def send_audio(self, audio_data: str):
        """Send audio data to Gemini"""
        logger.info("Preparing to send audio data to Gemini")
        realtime_input_msg = {
            "realtime_input": {
                "media_chunks": [
                    {
                        "data": audio_data,
                        "mime_type": "audio/pcm"
                    }
                ]
            }
        }
        try:
            await self.ws.send(json.dumps(realtime_input_msg))
            logger.info("Successfully sent audio data to Gemini")
        except Exception as e:
            logger.error(f"Failed to send audio data: {str(e)}", exc_info=True)
            raise

    async def send_frame(self, frame_data: str):
        """Send video frame data to Gemini"""
        logger.info("Preparing to send frame data to Gemini")
        realtime_input_msg = {
            "realtime_input": {
                "media_chunks": [
                    {
                        "data": frame_data,
                        "mime_type": "image/jpeg"
                    }
                ]
            }
        }
        try:
            await self.ws.send(json.dumps(realtime_input_msg))
            logger.info("Successfully sent frame data to Gemini")
        except Exception as e:
            logger.error(f"Failed to send frame data: {str(e)}", exc_info=True)
            raise

    async def receive(self):
        """Receive message from Gemini"""
        logger.info("Waiting for message from Gemini")
        try:
            response = await self.ws.recv()
            logger.info("Successfully received message from Gemini")
            return response
        except Exception as e:
            logger.error(f"Error receiving message from Gemini: {str(e)}", exc_info=True)
            raise

    async def close(self):
        """Close the connection"""
        if self.ws:
            logger.info("Closing Gemini connection")
            try:
                await self.ws.close()
                logger.info("Successfully closed Gemini connection")
            except Exception as e:
                logger.error(f"Error closing Gemini connection: {str(e)}", exc_info=True)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = id(websocket)
    logger.info(f"New websocket connection request from client {client_id}")
    await websocket.accept()
    logger.info(f"Accepted websocket connection from client {client_id}")
    
    try:
        logger.info(f"Creating Gemini connection for client {client_id}")
        gemini = GeminiConnection()
        
        logger.info(f"Initializing Gemini connection for client {client_id}")
        await gemini.connect()
        
        # Handle bidirectional communication
        async def receive_from_client():
            try:
                while True:
                    try:
                        data = await websocket.receive_json()
                        
                        if isinstance(data, dict):
                            if "realtimeInput" in data:
                                media_chunks = data["realtimeInput"]["mediaChunks"]
                                for chunk in media_chunks:
                                    if chunk["mimeType"].startswith("audio/"):
                                        await gemini.send_audio(chunk["data"])
                                    elif chunk["mimeType"].startswith("image/"):
                                        await gemini.send_frame(chunk["data"])
                            elif "clientContent" in data:
                                await gemini.send_text(data["clientContent"]["turns"][0]["parts"][0]["text"])
                            else:
                                logger.warning(f"Received invalid data format from client {client_id}")
                            
                    except json.JSONDecodeError:
                        logger.warning(f"Received invalid JSON from client {client_id}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error in receive_from_client for client {client_id}: {str(e)}", exc_info=True)

        async def receive_from_gemini():
            try:
                while True:
                    msg = await gemini.receive()
                    response = json.loads(msg)
                    logger.info(f"Received response from Gemini for client {client_id}")
                    # logger.debug(f"Response details: {json.dumps(response, indent=2)}")
                    
                    # Forward audio data to client
                    try:
                        parts = response["serverContent"]["modelTurn"]["parts"]
                        for p in parts:
                            if "inlineData" in p:
                                audio_data = p["inlineData"]["data"]
                                logger.info(f"Sending audio data to client {client_id}")
                                await websocket.send_json({
                                    "type": "audio",
                                    "data": audio_data
                                })
                            elif "text" in p:
                                text_content = p["text"]
                                logger.info(f"Sending text to client {client_id}: {text_content[:50]}...")
                                await websocket.send_json({
                                    "type": "text",
                                    "data": text_content
                                })
                    except KeyError:
                        logger.debug(f"No parts found in response for client {client_id}")

                    # Handle turn completion
                    try:
                        if response["serverContent"]["turnComplete"]:
                            logger.info(f"Turn complete for client {client_id}")
                            await websocket.send_json({
                                "type": "turn_complete",
                                "data": True
                            })
                    except KeyError:
                        pass
            except Exception as e:
                logger.error(f"Error in receive_from_gemini for client {client_id}: {str(e)}", exc_info=True)

        # Run both receiving tasks concurrently
        logger.info(f"Starting bidirectional communication for client {client_id}")
        async with asyncio.TaskGroup() as tg:
            tg.create_task(receive_from_client())
            tg.create_task(receive_from_gemini())

    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {str(e)}", exc_info=True)
    finally:
        # Cleanup
        logger.info(f"Cleaning up connection for client {client_id}")
        await gemini.close()
        logger.info(f"Connection cleanup complete for client {client_id}")

if __name__ == "__main__":
    logger.info("Starting server")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")