# BUILD
- set .env with gemini api key
- `docker compose up -d --build`

# API

Go to http://localhost:8008/docs to see all the routes.

There are some get routes to see the state of the database.

The main route you want to use is `/chat`. There are 3 actions you can do here: list sodas, list transactions, buy sodas.
The LLM will try to do the requested action and will also instruct the user if it can't understand his intentions.