module Server

using HTTP.WebSockets: WebSockets
using Sockets: Sockets
using JSON3: JSON3

using ..JackalControl: setup_trajectory_optimizer

function serve(; trajectory_optimizer = setup_trajectory_optimizer(), port = 8081, ip = "127.0.0.1")
    connection = WebSockets.listen!(ip, port) do ws
        # tune socket for performance
        Sockets.nagle(ws.io.io, false)
        Sockets.quickack(ws.io.io, true)
        for msg in ws
            request = JSON3.read(msg)
            result = trajectory_optimizer(request.state, request.goal, request.obstacle)
            WebSockets.send(ws, JSON3.write(result))
        end
    end
    println("press enter to stop the server")
    readline()
    WebSockets.forceclose(connection)
end

end
