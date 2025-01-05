# Copyright 2025 GlyphyAI
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "examples.chat_server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        timeout_keep_alive=120,
        timeout_graceful_shutdown=10,
        limit_concurrency=100,
        backlog=128,
        h11_max_incomplete_event_size=5_242_880,  # 5MB
    )
