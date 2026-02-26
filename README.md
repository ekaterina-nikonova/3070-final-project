# Web server

To start the web server, run the following command in your terminal from the project's root directory:

```shell
uvicorn web:app --host 0.0.0.0 --reload --app-dir src
```

The `--host 0.0.0.0` option is necessary to provide access to the server via the local network. Check the IP of the computer on which the server is running in the network settings, e.g. `192.168.84.51`, and test the connection in the command line. On macOS or Linux, you can run the following command in the terminal:

```shell
curl -v http://192.168.84.51:8000/help
```

To verify that the server is accessible, send a POST request to the `/generate-test` endpoint to receive a canned response containing a text and a list of questions:

```shell
curl -sS X POST -H "Content-Type: application/json" -d '{"topic": "<specify_the_topic_here>"}' http://192.168.84.51:8000/generate-test
```

In PowerShell on Windows, the `text` or the `questions` property of the server response must be used to avoid the truncation of the response in the console:

```shell
(Invoke-RestMethod -Uri "http://192.168.84.58:8000/generate-test" -Method POST -ContentType "application/json" -Body '{"topic": "<specify_the_topic_here>"}').text
```
