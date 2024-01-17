using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Revise, OhMyREPL, Infiltrator
using JackalControl

JackalControl.Server.serve()
