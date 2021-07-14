from werkzeug.wrappers import Request, Response
from dotenv import dotenv_values

config = dotenv_values(".env")


class Middleware:
    def __init__(self, app):
        self.app = app
        self.API_KEY = config["API_KEY"]

    def __call__(self, environ, start_response):
        try:
            # case: try authentication by API key
            request = Request(environ)
            if request.headers["X-Api-Key"] == self.API_KEY:
                # case: correct API_KEY
                return self.app(environ, start_response)
        except Exception:
            # case: failed authentication by API key
            print("Failed to authenticate")

        return Response("Failed to authenticate", status=401)(environ, start_response)
