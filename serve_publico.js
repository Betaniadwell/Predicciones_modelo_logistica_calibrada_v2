/**
 * Proxy inverso local: sirve la API y la web Streamlit tras un unico puerto.
 *
 *   /predict, /model-info, /health, /docs, /redoc  ->  API (FastAPI, 8000)
 *   todo lo demas                                   ->  Web (Streamlit, 8501)
 *
 * Motivo: la cuenta de ngrok solo admite un hostname, asi que un unico tunel
 * tiene que atender los dos servicios. El soporte de WebSocket es necesario
 * para que Streamlit funcione a traves del proxy.
 *
 * Uso:  node serve_publico.js      (ngrok http 8080)
 */
const http = require("http");
const net = require("net");

const API_PORT = Number(process.env.API_PORT || 8000);
const WEB_PORT = Number(process.env.WEB_PORT || 8501);
const PORT = Number(process.env.PORT || 8080);
const HOST = "127.0.0.1";

const RUTAS_API = [
  "/predict",
  "/model-info",
  "/health",
  "/docs",
  "/redoc",
  "/openapi.json",
];

function esApi(url) {
  const ruta = url.split("?")[0];
  return RUTAS_API.some(
    (p) => ruta === p || ruta.startsWith(p + "/")
  );
}

const servidor = http.createServer((req, res) => {
  const destino = esApi(req.url) ? API_PORT : WEB_PORT;
  const proxy = http.request(
    {
      host: HOST,
      port: destino,
      path: req.url,
      method: req.method,
      headers: req.headers,
    },
    (respuesta) => {
      res.writeHead(respuesta.statusCode, respuesta.headers);
      respuesta.pipe(res);
    }
  );
  proxy.on("error", (err) => {
    res.writeHead(502, { "Content-Type": "text/plain" });
    res.end("Proxy error: " + err.message);
  });
  req.pipe(proxy);
});

// WebSocket: lo usa Streamlit para el canal de eventos
servidor.on("upgrade", (req, socket, head) => {
  const proxy = net.connect(WEB_PORT, HOST, () => {
    let cabecera = `${req.method} ${req.url} HTTP/${req.httpVersion}\r\n`;
    for (let i = 0; i < req.rawHeaders.length; i += 2) {
      cabecera += `${req.rawHeaders[i]}: ${req.rawHeaders[i + 1]}\r\n`;
    }
    cabecera += "\r\n";
    proxy.write(cabecera);
    if (head && head.length) proxy.write(head);
    proxy.pipe(socket);
    socket.pipe(proxy);
  });
  proxy.on("error", () => socket.destroy());
  socket.on("error", () => proxy.destroy());
});

servidor.listen(PORT, HOST, () => {
  console.log(`Proxy escuchando en http://${HOST}:${PORT}`);
  console.log(`  API  -> http://${HOST}:${API_PORT}  (/predict, /docs, ...)`);
  console.log(`  Web  -> http://${HOST}:${WEB_PORT}  (Streamlit)`);
});
