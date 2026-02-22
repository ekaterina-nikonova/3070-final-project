# Web server

To start the web server, run the following command in your terminal from the project's root directory:

```shell
uvicorn web:app --host 0.0.0.0 --reload --app-dir src
```

Find the IP of the computer the server is running on. Test the connection in the command line (terminal or Power Shell on Windows):

```shell
curl -v http://192.168.84.51:8000/help
```

Send a POST request to the `/generate-test` endpoint to receive a canned response containing a text and a list of questions:

```shell
curl -sS X POST -H "Content-Type: application/json" -d '{"topic": "..."}' http://192.168.84.51:8000/generate-test
```

In PowerShell on Windows,

```shell
Invoke-RestMethod -Uri "http://192.168.84.58:8000/generate-test" -Method POST -ContentType "application/json" -Body '{"topic": "Sunday"}'
```
