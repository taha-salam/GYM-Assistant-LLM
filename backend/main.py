import json
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from conversation_manager import ConversationManager

app = FastAPI()

# session_id -> ConversationManager instance
sessions = {}


def get_or_create_session(session_id):
    if session_id and session_id in sessions:
        return session_id, sessions[session_id]

    new_id = session_id if session_id else str(uuid.uuid4())
    sessions[new_id] = ConversationManager()
    return new_id, sessions[new_id]


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()

    session_id = None
    manager = None

    try:
        while True:
            raw_data = await websocket.receive_text()

            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format. Expected a JSON object with a 'message' field."
                }))
                continue

            if not isinstance(data, dict) or "message" not in data:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Missing required 'message' field."
                }))
                continue

            user_message = data.get("message", "").strip()
            requested_session_id = data.get("session_id")

            if not user_message:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Message cannot be empty."
                }))
                continue

            if session_id is None:
                session_id, manager = get_or_create_session(requested_session_id)
                # Send back any existing history so the frontend can render it
                # after a page refresh or reconnect, since the backend keeps
                # the conversation in memory but the frontend's chat window
                # is just DOM state that gets wiped on reload.
                await websocket.send_text(json.dumps({
                    "type": "session_start",
                    "session_id": session_id,
                    "history": manager.history
                }))

            try:
                async for token in manager.stream_response_async(user_message):
                    await websocket.send_text(json.dumps({
                        "type": "token",
                        "content": token
                    }))
                await websocket.send_text(json.dumps({"type": "done"}))
            except WebSocketDisconnect:
                # The client disconnected mid-stream. Do not attempt to send
                # anything further on this socket, just let it propagate up
                # to the outer handler, which logs the disconnect cleanly.
                raise
            except Exception as e:
                # A real error (e.g. Ollama unreachable, malformed model
                # response). The socket is still open here, so it's safe
                # to report it back to the client.
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Something went wrong generating a response: {str(e)}"
                }))

    except WebSocketDisconnect:
        print(f"Session {session_id} disconnected, history preserved in memory.")