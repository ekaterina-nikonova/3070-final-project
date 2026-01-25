# Web server

To start the web server, run the following command in your terminal from the project's root directory:

```shell
uvicorn web:app --host 0.0.0.0 --reload --app-dir src
```

Find the IP of the computer the server is running on. Test the connection in the command line (terminal or Power Shell on Windows):

```shell
curl -v http://192.168.84.51:8000/help
```
