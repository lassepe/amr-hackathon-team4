using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

ENV["PATH_LICENSE_STRING"] = "2830898829&Courtesy&&&USR&45321&5_1_2021&1000&PATH&GEN&31_12_2025&0_0_0&6000&0_0"

using Revise, OhMyREPL, Infiltrator
using JackalControl

JackalControl.Server.serve()
