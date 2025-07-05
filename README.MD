# BUILD
- set .env with gemini api key
- `docker compose up -d --build`

# API

Go to http://localhost:8008/docs to see all the routes.

There are some get routes to see the state of the database.

The main route you want to use is `/chat`. There are 3 actions you can do here: list sodas, list transactions, buy sodas.
The LLM will try to do the requested action and will also instruct the user if it can't understand his intentions.

You can also ask him information about the data and since your message is passed to the prompt he might do something with it. Ex: "list me the sodas and also which is the cheapest soda you have?"

# Code
All the code is in the main.py (300 lines) file. a `database.db` file is already in the repo with some previous transactions, but you can delete it if you want to.

# Improvements
There are some possible improvements to make like error handling. It also would be nice to organize it into a proper project structure to continue development.