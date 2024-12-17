# -*- coding: utf-8 -*-
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import logging
import base64
import io
from typing import Dict
from google import genai

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
}

MODEL = f"models/{GEMINI_CONFIG['model']}"

CONFIG = {
    "generation_config": {
        "response_modalities": ["AUDIO"]
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

class AudioLoop:
    def __init__(self, session, websocket, client_id):
        self.session = session
        self.websocket = websocket
        self.client_id = client_id
        self.audio_in_queue = asyncio.Queue()
        self.audio_out_queue = asyncio.Queue()
        self.video_out_queue = asyncio.Queue()
        logger.info(f"Initialized AudioLoop with model: {GEMINI_CONFIG['model']}")

    async def send_text(self, text: str):
        """Send text input to Gemini"""
        try:
            await self.session.send(text, end_of_turn=True)
            logger.info("Successfully sent text to Gemini")
        except Exception as e:
            logger.error(f"Failed to send text: {str(e)}", exc_info=True)
            raise

    async def send_audio(self, audio_data: str):
        """Send audio data to Gemini"""
        logger.info("Preparing to send audio data to Gemini")
        try:
            decoded_data = base64.b64decode(audio_data)
            await self.session.send({"data": decoded_data, "mime_type": "audio/pcm"})
            logger.info("Successfully sent audio data to Gemini")
        except Exception as e:
            logger.error(f"Failed to send audio data: {str(e)}", exc_info=True)
            raise

    async def send_frame(self, frame_data: str):
        """Send video frame data to Gemini"""
        logger.info("Preparing to send frame data to Gemini")
        try:
            decoded_data = base64.b64decode(frame_data)
            await self.session.send({"data": decoded_data, "mime_type": "image/jpeg"})
            logger.info("Successfully sent frame data to Gemini")
        except Exception as e:
            logger.error(f"Failed to send frame data: {str(e)}", exc_info=True)
            raise

    async def receive_from_client(self):
        try:
            while True:
                try:
                    data = await self.websocket.receive_json()
                    
                    if isinstance(data, dict):
                        if "realtimeInput" in data:
                            media_chunks = data["realtimeInput"]["mediaChunks"]
                            for chunk in media_chunks:
                                if chunk["mimeType"].startswith("audio/"):
                                    await self.send_audio(chunk["data"])
                                elif chunk["mimeType"].startswith("image/"):
                                    await self.send_frame(chunk["data"])
                        elif "clientContent" in data:
                            await self.send_text(data["clientContent"]["turns"][0]["parts"][0]["text"])
                        else:
                            logger.warning(f"Received invalid data format from client {self.client_id}")
                        
                except json.JSONDecodeError:
                    logger.warning(f"Received invalid JSON from client {self.client_id}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in receive_from_client for client {self.client_id}: {str(e)}", exc_info=True)

    async def receive_from_gemini(self):
        try:
            while True:
                async for response in self.session.receive():
                    logger.info(f"Received response from Gemini for client {self.client_id}")
                    
                    server_content = response.server_content
                    if server_content is not None:
                        model_turn = server_content.model_turn
                        if model_turn is not None:
                            parts = model_turn.parts
                            
                            for part in parts:
                                if part.text is not None:
                                    logger.info(f"Sending text to client {self.client_id}: {part.text[:50]}...")
                                    await self.websocket.send_json({
                                        "type": "text",
                                        "data": part.text
                                    })
                                elif part.inline_data is not None:
                                    logger.info(f"Sending audio data to client {self.client_id}")
                                    audio_data = base64.b64encode(part.inline_data.data).decode('utf-8')
                                    await self.websocket.send_json({
                                        "type": "audio",
                                        "data": audio_data
                                    })
                            
                            server_content.model_turn = None
                            turn_complete = server_content.turn_complete
                            if turn_complete:
                                logger.info(f"Turn complete for client {self.client_id}")
                                await self.websocket.send_json({
                                    "type": "turn_complete",
                                    "data": True
                                })
                                # Clear audio queue on turn complete
                                while not self.audio_in_queue.empty():
                                    self.audio_in_queue.get_nowait()

        except Exception as e:
            logger.error(f"Error in receive_from_gemini for client {self.client_id}: {str(e)}", exc_info=True)

client = genai.Client(api_key=GEMINI_CONFIG["api_key"], http_options={'api_version': 'v1alpha'})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = id(websocket)
    logger.info(f"New websocket connection request from client {client_id}")
    await websocket.accept()
    logger.info(f"Accepted websocket connection from client {client_id}")
    
    logger.info(f"Creating Gemini connection for client {client_id}")
    
    async with (
        client.aio.live.connect(model=MODEL, config=CONFIG) as session,
        asyncio.TaskGroup() as tg,
    ):
        audio_loop = AudioLoop(session=session, websocket=websocket, client_id=client_id)
        
        # Run both receiving tasks concurrently
        logger.info(f"Starting bidirectional communication for client {client_id}")
        
        tasks = [
            tg.create_task(audio_loop.receive_from_client()),
            tg.create_task(audio_loop.receive_from_gemini())
        ]
        
        def check_error(task):
            if task.cancelled():
                return
            if task.exception() is not None:
                e = task.exception()
                logger.error(f"Task error: {str(e)}", exc_info=True)
                
        for task in tasks:
            task.add_done_callback(check_error)
            
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    logger.info("Starting server")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")